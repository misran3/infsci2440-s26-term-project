"""Streamlit web interface for Survey Analysis Agent."""

import asyncio
import logging

import altair as alt
from dotenv import load_dotenv
import pandas as pd
import streamlit as st

load_dotenv()

from src.classification.naive_bayes import TopicClassifier
from src.config import DATA, MODELS
from src.loaders.loader import load_reviews
from src.loaders.structures import Topic
from src.pipeline import PipelineComponents, SurveyAnalysisPipeline
from src.reasoning.bayesian_network import BayesianNetwork
from src.reasoning.hmm_sentiment import HMMSentiment
from src.reasoning.llm_summarizer import LLMSummarizer
from src.search.beam_search import BeamSearchExpander
from src.search.term_filter import TermFilter
from src.search.tfidf_retriever import TFIDFRetriever

logger = logging.getLogger(__name__)


def inject_custom_css() -> None:
    """Inject custom CSS for card styling."""
    st.markdown(
        """
        <style>
        /* Card container styling */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #e2e8f0;
            padding: 1rem;
            margin-bottom: 0.5rem;
            background: #ffffff;
        }

        /* Consistent spacing for result cards */
        .result-card {
            border: 1px solid #e2e8f0;
            padding: 1.5rem;
            margin-bottom: 1rem;
            background: #ffffff;
        }

        /* Subheader styling */
        .card-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: #1a202c;
        }

        /* Reduce chart container padding */
        div[data-testid="stVegaLiteChart"] {
            padding: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_model_metadata(metadata: dict[str, dict]) -> None:
    """Display model metadata in a structured table."""
    rows = []
    trained_times = []

    for model_name, meta in metadata.items():
        if not meta:
            rows.append({
                "Model": model_name,
                "Data Size": "N/A",
                "Parameters": "N/A",
                "Metrics": "N/A",
            })
            continue

        corpus_size = meta.get("corpus_size", "N/A")
        if isinstance(corpus_size, int):
            corpus_size = f"{corpus_size:,}"

        params = meta.get("params", {})
        metrics = meta.get("metrics", {})

        if model_name == "TF-IDF":
            params_str = f"max_features={params.get('max_features', 'N/A')}"
            metrics_str = f"vocab={metrics.get('vocabulary_size', 'N/A'):,}" if isinstance(metrics.get('vocabulary_size'), int) else "N/A"
        elif model_name == "Naive Bayes":
            params_str = f"alpha={params.get('alpha', 'N/A')}"
            classes = metrics.get("classes", [])
            metrics_str = f"{len(classes)} classes" if classes else "N/A"
        elif model_name == "Bayesian Net":
            structure = params.get("structure", [])
            if structure:
                params_str = "→".join([e[0][:3] for e in structure] + [structure[-1][1][:3]])
            else:
                params_str = "N/A"
            metrics_str = f"{metrics.get('n_cpds', 'N/A')} CPDs"
        elif model_name == "HMM":
            params_str = f"n_components={params.get('n_components', 'N/A')}"
            converged = metrics.get("converged")
            metrics_str = f"converged={converged}" if converged is not None else "N/A"
        else:
            params_str = str(params)[:30] if params else "N/A"
            metrics_str = str(metrics)[:30] if metrics else "N/A"

        rows.append({
            "Model": model_name,
            "Data Size": corpus_size,
            "Parameters": params_str,
            "Metrics": metrics_str,
        })

        if meta.get("trained_at"):
            trained_times.append(meta["trained_at"])

    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)

    if trained_times:
        latest = max(trained_times)
        trained_display = latest[:16].replace("T", " ")
        st.caption(f"Trained: {trained_display}")


st.set_page_config(
    page_title="Survey Analysis Agent",
    page_icon="🔍",
    layout="wide",
)

inject_custom_css()


@st.cache_resource
def load_pipeline() -> tuple[SurveyAnalysisPipeline, dict[str, dict]]:
    """Load all models and create pipeline (cached)."""
    metadata = {}

    with st.spinner("Loading models..."):
        # Load full corpus for TF-IDF retrieval
        reviews = load_reviews(path=DATA.clean_reviews, sample=False)

        # Initialize components
        expander = BeamSearchExpander(beam_width=3, max_depth=2)

        # Load TF-IDF from disk (trained on full corpus)
        from src.search.tfidf_retriever import load_retriever
        retriever, tfidf_meta = load_retriever(str(MODELS.tfidf_vectorizer), reviews)
        metadata["TF-IDF"] = tfidf_meta

        classifier = TopicClassifier.load(str(MODELS.naive_bayes))
        metadata["Naive Bayes"] = getattr(classifier, "metadata", {})

        try:
            bayesian_net = BayesianNetwork.load(MODELS.bayesian_network)
            metadata["Bayesian Net"] = getattr(bayesian_net, "metadata", {})
            logger.info("Loaded trained Bayesian Network")
        except FileNotFoundError:
            logger.warning("Bayesian model not found, using unfitted instance")
            bayesian_net = BayesianNetwork()
            metadata["Bayesian Net"] = {}

        try:
            hmm = HMMSentiment.load(MODELS.hmm_model)
            metadata["HMM"] = getattr(hmm, "metadata", {})
            logger.info("Loaded trained HMM model")
        except FileNotFoundError:
            logger.warning("HMM model not found, using unfitted instance")
            hmm = HMMSentiment()
            metadata["HMM"] = {}

        summarizer = LLMSummarizer()

        # Try to create TermFilter, fallback to None if AWS not configured
        try:
            term_filter = TermFilter()
            logger.info("TermFilter initialized successfully")
        except Exception as e:
            logger.warning(f"TermFilter unavailable: {e}")
            term_filter = None

        components = PipelineComponents(
            expander=expander,
            retriever=retriever,
            classifier=classifier,
            bayesian_net=bayesian_net,
            hmm=hmm,
            summarizer=summarizer,
            term_filter=term_filter,
        )

        return SurveyAnalysisPipeline(components), metadata


def main():
    st.title("Survey Analysis Agent")
    st.markdown("*Analyze product reviews using AI techniques from INFSCI2440*")

    # Load pipeline
    try:
        pipeline, metadata = load_pipeline()
        st.success("Models loaded successfully")
        with st.expander("Model Information", expanded=False):
            display_model_metadata(metadata)
    except FileNotFoundError:
        st.error("Model not found! Please run: `uv run python scripts/train_classifier.py --dataset sample`")
        return

    # Query input
    st.markdown("---")
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., app crashes, easy to use, too expensive",
        )

    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        search_clicked = st.button("Search", type="primary", width="stretch")

    # Optional filters
    with st.expander("Optional Filters"):
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            topic_options = ["Auto-detect"] + [t.value for t in Topic]
            topic_filter = st.selectbox("Topic Filter", topic_options)

        with filter_col2:
            min_confidence = st.slider(
                "Min Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
            )

        top_k = st.slider(
            "TF-IDF Candidates",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            help="Number of candidate reviews to retrieve before filtering",
        )

    # Run pipeline
    if search_clicked and query:
        run_pipeline_and_display(
            pipeline,
            query,
            None if topic_filter == "Auto-detect" else topic_filter,
            min_confidence,
            top_k,
        )


def run_pipeline_uncached(
    pipeline: SurveyAnalysisPipeline,
    query: str,
    topic_filter: str | None,
    min_confidence: float,
    top_k: int = 500,
):
    """Run pipeline (uncached - results contain complex objects)."""
    return asyncio.run(pipeline.run(query, topic_filter, min_confidence, top_k))


def _show_topic_distribution_chart(distribution: dict[str, int]) -> None:
    """Display topic distribution as a horizontal bar chart."""
    if not distribution:
        st.caption("No topics found")
        return

    df = pd.DataFrame(
        sorted(distribution.items(), key=lambda x: x[1], reverse=True),
        columns=["Topic", "Count"],
    )

    chart = alt.Chart(df).mark_bar(color="#2c5282").encode(
        x=alt.X("Count:Q", title="Count"),
        y=alt.Y("Topic:N", sort="-x", title=""),
        tooltip=["Topic", "Count"]
    ).properties(
        height=150
    )
    st.altair_chart(chart, use_container_width=False)


def _sentiment_to_emoji(sentiment) -> str:
    """Map sentiment state to colored emoji."""
    mapping = {
        "positive": "🟢",
        "negative": "🔴",
        "neutral": "🟡",
    }
    return mapping.get(sentiment.value if hasattr(sentiment, 'value') else str(sentiment), "⚪")


def _rating_to_stars(rating: int) -> str:
    """Convert numeric rating to star display."""
    filled = "★" * rating
    empty = "☆" * (5 - rating)
    return filled + empty


def _aggregate_hmm_transitions(sequences: list) -> pd.DataFrame:
    """Aggregate transition probabilities across all sequences into a matrix."""
    if not sequences:
        return pd.DataFrame()

    transition_keys = [
        "pos_to_pos", "pos_to_neg", "pos_to_neu",
        "neg_to_pos", "neg_to_neg", "neg_to_neu",
        "neu_to_pos", "neu_to_neg", "neu_to_neu",
    ]

    # Sum transitions across all sequences
    totals = {k: 0.0 for k in transition_keys}
    count = 0
    for seq in sequences:
        if seq.transitions:
            for k in transition_keys:
                totals[k] += seq.transitions.get(k, 0.0)
            count += 1

    if count == 0:
        return pd.DataFrame()

    # Average
    avg = {k: totals[k] / count for k in transition_keys}

    # Build matrix format for heatmap
    rows = []
    for from_state in ["Positive", "Negative", "Neutral"]:
        for to_state in ["Positive", "Negative", "Neutral"]:
            key = f"{from_state[:3].lower()}_to_{to_state[:3].lower()}"
            rows.append({
                "From": from_state,
                "To": to_state,
                "Probability": avg.get(key, 0.0)
            })

    return pd.DataFrame(rows)


def run_pipeline_and_display(
    pipeline: SurveyAnalysisPipeline,
    query: str,
    topic_filter: str | None,
    min_confidence: float,
    top_k: int = 500,
):
    """Run pipeline and display results."""
    with st.spinner("Analyzing..."):
        result, filter_result = run_pipeline_uncached(
            pipeline, query, topic_filter, min_confidence, top_k
        )

    st.markdown("---")
    st.header("Pipeline Results")

    # 1. Query Expansion
    with st.container():
        st.subheader("1. Query Expansion (Beam Search + LLM Filter)")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Original:**")
        with col2:
            st.code(result.expansion.original_query)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**BeamSearch:**")
        with col2:
            st.markdown(f"{len(result.expansion.expanded_terms)} terms")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**LLM Filtered:**")
        with col2:
            if len(result.filtered_terms) < len(result.expansion.expanded_terms):
                st.markdown(f"{len(result.filtered_terms)} terms")
            elif pipeline.components.term_filter is None:
                st.markdown(f"{len(result.filtered_terms)} terms (LLM filter skipped - AWS not configured)")
            else:
                st.markdown(f"{len(result.filtered_terms)} terms (no terms removed)")

        with st.expander("Show expansion details"):
            st.markdown("**BeamSearch terms:**")
            st.code(", ".join(result.expansion.expanded_terms))

            st.markdown("**Filtered terms (used for retrieval):**")
            st.code(", ".join(result.filtered_terms))

            removed = set(result.expansion.expanded_terms) - set(result.filtered_terms)
            if removed:
                st.markdown("**Removed terms:**")
                st.code(", ".join(sorted(removed)))

            st.markdown("**Beam paths:**")
            for path in result.expansion.beam_paths[:5]:
                st.text(f"  {' → '.join(path['path'])} (score: {path['score']:.2f})")

    # 2. TF-IDF Retrieval
    with st.container():
        st.subheader("2. TF-IDF Retrieval")
        st.markdown(f"Found **{len(result.candidate_reviews)}** candidate reviews")

        with st.expander("Show sample reviews"):
            for review in result.candidate_reviews[:5]:
                stars = "⭐" * review.rating
                score_str = f"TF-IDF relevance: {review.tfidf_score:.3f}" if review.tfidf_score else ""
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{review.title}** {stars}")
                with col2:
                    st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.85em;'>{score_str}</div>", unsafe_allow_html=True)
                st.caption(review.text[:200] + "..." if len(review.text) > 200 else review.text)
                st.divider()

    # 3. Topic Classification
    with st.container():
        st.subheader("3. Topic Classification (Naive Bayes)")

        detected_topic = topic_filter or pipeline.components.classifier.detect_topic_from_query(query)
        st.markdown(f"Detected topic: **{detected_topic}**")

        if filter_result.fallback_used:
            # Count how many were classified as target topic (before confidence filter)
            topic_count = filter_result.topic_distribution.get(detected_topic, 0)
            if topic_count > 0:
                st.warning(
                    f"⚠️ {topic_count} reviews classified as \"{detected_topic}\" "
                    f"but none met classification confidence threshold ({min_confidence:.0%})"
                )
                st.info("💡 Try lowering the confidence threshold in Optional Filters, or retrain on full dataset")
            else:
                st.warning(f"⚠️ No reviews matched topic \"{detected_topic}\"")
            st.markdown("**Topics found in search results:**")
            _show_topic_distribution_chart(filter_result.topic_distribution)
        else:
            st.markdown(f"Filtered to **{len(result.filtered_reviews)}** relevant reviews")

            st.markdown("**Topic distribution:**")
            _show_topic_distribution_chart(filter_result.topic_distribution)

            with st.expander("Show filtered reviews"):
                for review, classification in zip(
                    result.filtered_reviews[:5],
                    filter_result.classifications[:5],
                ):
                    stars = "⭐" * review.rating
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{review.title}** {stars}")
                    with col2:
                        st.markdown(
                            f"<div style='text-align: right; color: gray; font-size: 0.85em;'>"
                            f"confidence: {classification.confidence:.1%}</div>",
                            unsafe_allow_html=True,
                        )
                    st.caption(review.text[:200] + "..." if len(review.text) > 200 else review.text)
                    st.divider()

    # 4. Bayesian Network
    with st.container():
        st.subheader("4. Probabilistic Insights (Bayesian Network)")
        insights = result.bayesian_insights

        if insights.p_negative_given_topic == 0.0 and insights.p_positive_given_topic == 0.0:
            st.info("Bayesian Network not yet fitted")
        else:
            col1, col2 = st.columns(2)

            # Left chart: Sentiment | Topic
            with col1:
                p_neutral = max(0, 1 - insights.p_positive_given_topic - insights.p_negative_given_topic)
                sentiment_data = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Probability": [
                        insights.p_positive_given_topic,
                        insights.p_negative_given_topic,
                        p_neutral
                    ],
                    "Color": ["#2ecc71", "#e74c3c", "#f39c12"]
                })

                chart1 = alt.Chart(sentiment_data).mark_bar().encode(
                    x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
                    y=alt.Y("Sentiment:N", sort=["Positive", "Negative", "Neutral"], title=""),
                    color=alt.Color("Color:N", scale=None),
                    tooltip=["Sentiment", alt.Tooltip("Probability:Q", format=".1%")]
                ).properties(
                    title=f"Sentiment | Topic: {insights.topic.value}",
                    height=120
                )
                st.altair_chart(chart1, use_container_width=False)

            # Right chart: Rating | Sentiment
            with col2:
                rating_data = pd.DataFrame({
                    "Condition": ["High Rating | Positive", "Low Rating | Negative"],
                    "Probability": [
                        insights.p_high_rating_given_positive,
                        insights.p_low_rating_given_negative
                    ],
                    "Color": ["#2ecc71", "#e74c3c"]
                })

                chart2 = alt.Chart(rating_data).mark_bar().encode(
                    x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
                    y=alt.Y("Condition:N", title=""),
                    color=alt.Color("Color:N", scale=None),
                    tooltip=["Condition", alt.Tooltip("Probability:Q", format=".1%")]
                ).properties(
                    title="Rating | Sentiment",
                    height=120
                )
                st.altair_chart(chart2, use_container_width=False)

    # 5. HMM Sentiment
    with st.container():
        st.subheader("5. Sentiment Flow (HMM)")

        sequences = result.sentiment_sequences
        valid_sequences = [s for s in sequences if s.sentiment_states]

        if not valid_sequences:
            st.info("HMM Sentiment Analysis: No sequences with sentiment states")
        else:
            # Aggregate stats
            avg_sentences = sum(len(s.sentences) for s in valid_sequences) / len(valid_sequences)
            st.markdown(f"**Analyzed {len(valid_sequences)} reviews**, average {avg_sentences:.1f} sentences per review")

            # Transition heatmap
            transition_df = _aggregate_hmm_transitions(valid_sequences)
            if not transition_df.empty:
                heatmap = alt.Chart(transition_df).mark_rect().encode(
                    x=alt.X("To:N", title="To State", sort=["Positive", "Negative", "Neutral"]),
                    y=alt.Y("From:N", title="From State", sort=["Positive", "Negative", "Neutral"]),
                    color=alt.Color(
                        "Probability:Q",
                        scale=alt.Scale(scheme="blues"),
                        title="Probability"
                    ),
                    tooltip=[
                        alt.Tooltip("From:N"),
                        alt.Tooltip("To:N"),
                        alt.Tooltip("Probability:Q", format=".1%")
                    ]
                ).properties(
                    title="Sentiment Transition Probabilities (averaged across reviews)",
                    width=300,
                    height=200
                )

                # Add text labels on cells
                text = alt.Chart(transition_df).mark_text(baseline="middle").encode(
                    x=alt.X("To:N", sort=["Positive", "Negative", "Neutral"]),
                    y=alt.Y("From:N", sort=["Positive", "Negative", "Neutral"]),
                    text=alt.Text("Probability:Q", format=".0%"),
                    color=alt.condition(
                        alt.datum.Probability > 0.5,
                        alt.value("white"),
                        alt.value("black")
                    )
                )

                st.altair_chart(heatmap + text, width="stretch")

            # Sample reviews with sentiment timeline
            multi_sentence = [s for s in valid_sequences if len(s.sentences) > 1][:5]
            if multi_sentence:
                with st.expander(f"Sample Reviews ({len(multi_sentence)})"):
                    for seq in multi_sentence:
                        # Find matching review for rating
                        matching_review = next(
                            (r for r in result.filtered_reviews if r.review_id == seq.review_id),
                            None
                        )
                        rating_str = _rating_to_stars(matching_review.rating) if matching_review else ""

                        # Truncate text
                        review_text = matching_review.text if matching_review else ""
                        truncated = review_text[:100] + "..." if len(review_text) > 100 else review_text

                        # Sentiment timeline
                        timeline = "".join(_sentiment_to_emoji(s) for s in seq.sentiment_states)

                        st.markdown(f'"{truncated}" {rating_str}')
                        st.caption(f"Sentiment flow: {timeline}")
                        st.markdown("---")

    # 6. LLM Summary
    with st.container():
        st.subheader("6. Summary (LLM)")

        # Main summary
        st.markdown(result.llm_summary)

        # Key themes (if available)
        if result.llm_themes:
            st.markdown("**Key Themes:**")
            for theme in result.llm_themes:
                st.markdown(f"- {theme}")

        # Representative quotes (if available)
        if result.llm_quotes:
            st.markdown("**Representative Quotes:**")
            for quote in result.llm_quotes:
                st.markdown(f'> "{quote}"')


if __name__ == "__main__":
    main()
