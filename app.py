"""Streamlit web interface for Survey Analysis Agent."""

import asyncio
import logging

import streamlit as st

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


st.set_page_config(
    page_title="Survey Analysis Agent",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def load_pipeline() -> SurveyAnalysisPipeline:
    """Load all models and create pipeline (cached)."""
    with st.spinner("Loading models..."):
        # Load corpus
        reviews = load_reviews(path=DATA.clean_reviews, limit=50000)

        # Initialize components
        expander = BeamSearchExpander(beam_width=3, max_depth=2)

        retriever = TFIDFRetriever(reviews)
        retriever.fit()

        classifier = TopicClassifier.load(str(MODELS.naive_bayes))

        try:
            bayesian_net = BayesianNetwork.load(MODELS.bayesian_network)
            logger.info("Loaded trained Bayesian Network")
        except FileNotFoundError:
            logger.warning("Bayesian model not found, using unfitted instance")
            bayesian_net = BayesianNetwork()

        try:
            hmm = HMMSentiment.load(MODELS.hmm_model)
            logger.info("Loaded trained HMM model")
        except FileNotFoundError:
            logger.warning("HMM model not found, using unfitted instance")
            hmm = HMMSentiment()
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

        return SurveyAnalysisPipeline(components)


def main():
    st.title("🔍 Survey Analysis Agent")
    st.markdown("*Analyze product reviews using AI techniques from INFSCI2440*")

    # Load pipeline
    try:
        pipeline = load_pipeline()
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
        search_clicked = st.button("Search", type="primary", use_container_width=True)

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

    # Run pipeline
    if search_clicked and query:
        run_pipeline_and_display(
            pipeline,
            query,
            None if topic_filter == "Auto-detect" else topic_filter,
            min_confidence,
        )


def run_pipeline_uncached(
    pipeline: SurveyAnalysisPipeline,
    query: str,
    topic_filter: str | None,
    min_confidence: float,
):
    """Run pipeline (uncached - results contain complex objects)."""
    return asyncio.run(pipeline.run(query, topic_filter, min_confidence))


def _show_topic_distribution_chart(distribution: dict[str, int]) -> None:
    """Display topic distribution as a horizontal bar chart."""
    import pandas as pd

    if not distribution:
        st.caption("No topics found")
        return

    df = pd.DataFrame(
        sorted(distribution.items(), key=lambda x: x[1], reverse=True),
        columns=["Topic", "Count"],
    )
    st.bar_chart(df, x="Topic", y="Count")


def run_pipeline_and_display(
    pipeline: SurveyAnalysisPipeline,
    query: str,
    topic_filter: str | None,
    min_confidence: float,
):
    """Run pipeline and display results."""
    with st.spinner("Analyzing..."):
        result, filter_result = run_pipeline_uncached(
            pipeline, query, topic_filter, min_confidence
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

        if insights.p_negative_given_topic == 0.0:
            st.info("Bayesian Network not yet implemented (Person 3)")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "P(Negative | Topic)",
                    f"{insights.p_negative_given_topic:.0%}",
                )
            with col2:
                st.metric(
                    "P(Low Rating | Negative)",
                    f"{insights.p_low_rating_given_negative:.0%}",
                )

    # 5. HMM Sentiment
    with st.container():
        st.subheader("5. Sentiment Sequences (HMM)")

        if not result.sentiment_sequences or not result.sentiment_sequences[0].sentiment_states:
            st.info("HMM Sentiment Analysis not yet implemented (Person 3)")
        else:
            st.markdown("Common sentiment patterns in reviews")

    # 6. LLM Summary
    with st.container():
        st.subheader("6. Summary (LLM)")
        st.markdown(result.llm_summary)


if __name__ == "__main__":
    main()
