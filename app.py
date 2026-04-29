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
from src.search.query_preprocessor import QueryPreprocessor
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

        preprocessor = QueryPreprocessor()

        components = PipelineComponents(
            expander=expander,
            retriever=retriever,
            classifier=classifier,
            bayesian_net=bayesian_net,
            hmm=hmm,
            summarizer=summarizer,
            term_filter=term_filter,
            preprocessor=preprocessor,
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
    st.altair_chart(chart, width="content")


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


def _display_llm_summary_card(result) -> None:
    """Display LLM Summary card content."""
    st.markdown("#### Summary")
    st.markdown(result.llm_summary)

    if result.llm_themes:
        st.markdown("**Key Themes:**")
        for theme in result.llm_themes:
            st.markdown(f"- {theme}")

    if result.llm_quotes:
        st.markdown("**Representative Quotes:**")
        for quote in result.llm_quotes:
            st.markdown(f'> "{quote}"')


def _display_query_expansion_card(result, pipeline) -> None:
    """Display Query Expansion card content."""
    # Show preprocessing if it happened
    if result.preprocessed and result.preprocessed.was_preprocessed:
        st.markdown("#### Query Preprocessing")
        st.markdown("**Natural Language Query Detected**")
        st.markdown(f"**Original:** `{result.preprocessed.original_query}`")
        st.markdown(f"**Extracted Keywords:** `{', '.join(result.preprocessed.extracted_keywords)}`")
        st.divider()

    st.markdown("#### Query Expansion")
    st.markdown(f"**Original:** `{result.expansion.original_query}`")
    st.markdown(f"**BeamSearch:** {len(result.expansion.expanded_terms)} terms")

    if len(result.filtered_terms) < len(result.expansion.expanded_terms):
        st.markdown(f"**LLM Filtered:** {len(result.filtered_terms)} terms")
    elif pipeline.components.term_filter is None:
        st.markdown(f"**LLM Filtered:** {len(result.filtered_terms)} terms *(skipped)*")
    else:
        st.markdown(f"**LLM Filtered:** {len(result.filtered_terms)} terms *(no removal)*")

    with st.expander("Details"):
        st.markdown("**BeamSearch terms:**")
        st.code(", ".join(result.expansion.expanded_terms))
        st.markdown("**Filtered terms:**")
        st.code(", ".join(result.filtered_terms))
        removed = set(result.expansion.expanded_terms) - set(result.filtered_terms)
        if removed:
            st.markdown("**Removed:**")
            st.code(", ".join(sorted(removed)))
        st.markdown("**Beam paths:**")
        for path in result.expansion.beam_paths[:5]:
            st.text(f"  {' → '.join(path['path'])} (score: {path['score']:.2f})")


def _display_tfidf_card(result) -> None:
    """Display TF-IDF Retrieval card content."""
    st.markdown("#### TF-IDF Retrieval")
    st.markdown(f"Found **{len(result.candidate_reviews)}** candidates")

    with st.expander("Sample reviews"):
        for review in result.candidate_reviews[:5]:
            stars = "⭐" * review.rating
            score_str = f"TF-IDF: {review.tfidf_score:.3f}" if review.tfidf_score else ""
            st.markdown(f"**{review.title}** {stars}")
            if score_str:
                st.caption(score_str)
            st.caption(review.text[:200] + "..." if len(review.text) > 200 else review.text)
            st.divider()


def _display_topic_classification_card(result, filter_result, pipeline, query, topic_filter, min_confidence) -> None:
    """Display Topic Classification card content."""
    st.markdown("#### Topic Classification")
    detected_topic = topic_filter or pipeline.components.classifier.detect_topic_from_query(query)
    st.markdown(f"Detected: **{detected_topic}**")

    if filter_result.fallback_used:
        topic_count = filter_result.topic_distribution.get(detected_topic, 0)
        if topic_count > 0:
            st.warning(f"⚠️ {topic_count} reviews below confidence ({min_confidence:.0%})")
        else:
            st.warning(f"⚠️ No reviews matched \"{detected_topic}\"")
    else:
        st.markdown(f"Filtered to **{len(result.filtered_reviews)}** reviews")

    _show_topic_distribution_chart(filter_result.topic_distribution)

    if not filter_result.fallback_used and result.filtered_reviews:
        with st.expander("Filtered reviews"):
            for review, classification in zip(
                result.filtered_reviews[:5],
                filter_result.classifications[:5],
            ):
                stars = "⭐" * review.rating
                st.markdown(f"**{review.title}** {stars}")
                st.caption(f"Confidence: {classification.confidence:.1%}")
                st.caption(review.text[:200] + "..." if len(review.text) > 200 else review.text)
                st.divider()


def _display_bayesian_card(result) -> None:
    """Display Bayesian Network card content."""
    st.markdown("#### Bayesian Network")
    insights = result.bayesian_insights

    if insights.p_negative_given_topic == 0.0 and insights.p_positive_given_topic == 0.0:
        st.info("Not yet fitted")
        return

    # Sentiment | Topic chart
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
        height=150
    )
    st.altair_chart(chart1, width="stretch")

    # Rating | Sentiment chart
    rating_data = pd.DataFrame({
        "Condition": ["High Rating | Pos", "Low Rating | Neg"],
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
        height=150
    )
    st.altair_chart(chart2, width="stretch")


def _display_hmm_card(result) -> None:
    """Display HMM Sentiment Flow card content."""
    st.markdown("#### Sentiment Flow (HMM)")

    sequences = result.sentiment_sequences
    valid_sequences = [s for s in sequences if s.sentiment_states]

    if not valid_sequences:
        st.info("No sequences with sentiment states")
        return

    avg_sentences = sum(len(s.sentences) for s in valid_sequences) / len(valid_sequences)
    st.markdown(f"**{len(valid_sequences)} reviews**, avg {avg_sentences:.1f} sentences")

    transition_df = _aggregate_hmm_transitions(valid_sequences)
    if not transition_df.empty:
        heatmap = alt.Chart(transition_df).mark_rect().encode(
            x=alt.X("To:N", title="To", sort=["Positive", "Negative", "Neutral"]),
            y=alt.Y("From:N", title="From", sort=["Positive", "Negative", "Neutral"]),
            color=alt.Color("Probability:Q", scale=alt.Scale(scheme="blues"), title="Prob"),
            tooltip=[
                alt.Tooltip("From:N"),
                alt.Tooltip("To:N"),
                alt.Tooltip("Probability:Q", format=".1%")
            ]
        ).properties(width=280, height=180)

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
        st.altair_chart(heatmap + text, width="content")

    multi_sentence = [s for s in valid_sequences if len(s.sentences) > 1][:5]
    if multi_sentence:
        with st.expander(f"Samples ({len(multi_sentence)})"):
            for seq in multi_sentence:
                matching_review = next(
                    (r for r in result.filtered_reviews if r.review_id == seq.review_id),
                    None
                )
                rating_str = _rating_to_stars(matching_review.rating) if matching_review else ""
                review_text = matching_review.text if matching_review else ""
                truncated = review_text[:100] + "..." if len(review_text) > 100 else review_text
                timeline = "".join(_sentiment_to_emoji(s) for s in seq.sentiment_states)
                st.markdown(f'"{truncated}" {rating_str}')
                st.caption(f"Flow: {timeline}")
                st.markdown("---")


def run_pipeline_and_display(
    pipeline: SurveyAnalysisPipeline,
    query: str,
    topic_filter: str | None,
    min_confidence: float,
    top_k: int = 500,
):
    """Run pipeline and display results in card grid layout."""
    with st.spinner("Analyzing..."):
        result, filter_result = run_pipeline_uncached(
            pipeline, query, topic_filter, min_confidence, top_k
        )

    st.markdown("---")
    st.header("Results")

    # Row 1: LLM Summary | Query Expansion
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        with st.container(border=True, height=400):
            _display_llm_summary_card(result)
    with row1_col2:
        with st.container(border=True, height=400):
            _display_query_expansion_card(result, pipeline)

    # Row 2: TF-IDF | Topic Classification
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        with st.container(border=True, height=350):
            _display_tfidf_card(result)
    with row2_col2:
        with st.container(border=True, height=350):
            _display_topic_classification_card(result, filter_result, pipeline, query, topic_filter, min_confidence)

    # Row 3: Bayesian | HMM
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        with st.container(border=True, height=380):
            _display_bayesian_card(result)
    with row3_col2:
        with st.container(border=True, height=380):
            _display_hmm_card(result)


if __name__ == "__main__":
    main()
