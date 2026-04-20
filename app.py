"""Streamlit web interface for Survey Analysis Agent."""

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
from src.search.tfidf_retriever import TFIDFRetriever


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

        bayesian_net = BayesianNetwork()
        hmm = HMMSentiment()
        summarizer = LLMSummarizer()

        components = PipelineComponents(
            expander=expander,
            retriever=retriever,
            classifier=classifier,
            bayesian_net=bayesian_net,
            hmm=hmm,
            summarizer=summarizer,
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


@st.cache_data
def run_cached_pipeline(
    _pipeline: SurveyAnalysisPipeline,
    query: str,
    topic_filter: str | None,
    min_confidence: float,
):
    """Run pipeline with caching."""
    return _pipeline.run(query, topic_filter, min_confidence)


def run_pipeline_and_display(
    pipeline: SurveyAnalysisPipeline,
    query: str,
    topic_filter: str | None,
    min_confidence: float,
):
    """Run pipeline and display results."""
    with st.spinner("Analyzing..."):
        result, filter_result = run_cached_pipeline(
            pipeline, query, topic_filter, min_confidence
        )

    st.markdown("---")
    st.header("Pipeline Results")

    # 1. Query Expansion
    with st.container():
        st.subheader("1. Query Expansion (Beam Search)")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Original:**")
        with col2:
            st.code(result.expansion.original_query)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Expanded:**")
        with col2:
            st.code(", ".join(result.expansion.expanded_terms))

        with st.expander("Show beam paths"):
            for path in result.expansion.beam_paths[:5]:
                st.text(f"  {' → '.join(path['path'])} (score: {path['score']:.2f})")

    # 2. TF-IDF Retrieval
    with st.container():
        st.subheader("2. TF-IDF Retrieval")
        st.markdown(f"Found **{len(result.candidate_reviews)}** candidate reviews")

    # 3. Topic Classification
    with st.container():
        st.subheader("3. Topic Classification (Naive Bayes)")

        detected_topic = topic_filter or pipeline.components.classifier.detect_topic_from_query(query)
        st.markdown(f"Detected topic: **{detected_topic}**")

        if filter_result.fallback_used:
            st.warning(f"⚠️ No reviews matched topic \"{detected_topic}\"")
            st.markdown("**Topics found in search results:**")
            for topic, count in sorted(
                filter_result.topic_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                if count > 0:
                    st.markdown(f"  • {topic}: {count} reviews")
        else:
            st.markdown(f"Filtered to **{len(result.filtered_reviews)}** relevant reviews")

            st.markdown("**Topic distribution:**")
            for topic, count in sorted(
                filter_result.topic_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]:
                st.markdown(f"  • {topic}: {count}")

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
