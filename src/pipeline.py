"""Main pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

from src.classification.naive_bayes import TopicClassifier
from src.loaders.structures import (
    FilterResult,
    PipelineResult,
    QueryExpansion,
    Review,
)
from src.reasoning.bayesian_network import BayesianNetwork
from src.reasoning.hmm_sentiment import HMMSentiment
from src.reasoning.llm_summarizer import LLMSummarizer
from src.search.beam_search import BeamSearchExpander
from src.search.term_filter import TermFilter
from src.search.tfidf_retriever import TFIDFRetriever


@dataclass
class PipelineComponents:
    """Container for pipeline components."""
    expander: BeamSearchExpander
    retriever: TFIDFRetriever
    classifier: TopicClassifier
    bayesian_net: BayesianNetwork
    hmm: HMMSentiment
    summarizer: LLMSummarizer
    term_filter: TermFilter | None = None


class SurveyAnalysisPipeline:
    """Orchestrates the full analysis pipeline."""

    def __init__(self, components: PipelineComponents) -> None:
        self.components = components

    async def run(
        self,
        query: str,
        topic_filter: str | None = None,
        min_confidence: float = 0.5,
        top_k: int = 500,
    ) -> tuple[PipelineResult, FilterResult]:
        """
        Run the full pipeline.

        Args:
            query: User's search query.
            topic_filter: Optional topic to filter by (auto-detected if None).
            min_confidence: Minimum classification confidence.
            top_k: Number of TF-IDF candidates to retrieve.

        Returns:
            Tuple of (PipelineResult, FilterResult).
        """
        # 1. Expand query (Person 1 - implemented)
        expansion = self.components.expander.expand(query)

        # 2. Filter terms (LLM - async, optional)
        if self.components.term_filter:
            filtered_terms = await self.components.term_filter.filter(
                query, expansion.expanded_terms
            )
        else:
            filtered_terms = list(expansion.expanded_terms)

        # 3. TF-IDF retrieve (Person 1 - implemented)
        candidates = self.components.retriever.retrieve(
            filtered_terms, top_k=top_k
        )

        # 4. Classify + filter (Person 2)
        topic = topic_filter or self.components.classifier.detect_topic_from_query(query)
        filter_result = self.components.classifier.filter_by_topic(
            candidates, topic, min_confidence
        )

        # Use filtered reviews if available, else fall back to candidates
        reviews_for_analysis = filter_result.filtered_reviews or candidates

        # 5. Bayesian inference (Person 3 - stub)
        insights = self.components.bayesian_net.infer(reviews_for_analysis, topic)

        # 6. HMM analysis (Person 3 - stub)
        sequences = self.components.hmm.analyze(reviews_for_analysis)

        # 7. LLM summary (Person 3 - stub)
        summary = self.components.summarizer.summarize(
            reviews_for_analysis, insights, sequences
        )

        pipeline_result = PipelineResult(
            query=query,
            expansion=expansion,
            filtered_terms=filtered_terms,
            candidate_reviews=candidates,
            filtered_reviews=reviews_for_analysis,
            topic_classifications=filter_result.classifications,
            bayesian_insights=insights,
            sentiment_sequences=sequences,
            llm_summary=summary,
        )

        return pipeline_result, filter_result
