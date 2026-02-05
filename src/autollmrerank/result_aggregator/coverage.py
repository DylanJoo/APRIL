"""
Coverage-based result aggregation.

Coverage aggregation selects documents that maximize coverage of all sub-queries
(query aspects). This ensures the final ranking covers diverse aspects of the
original complex query rather than over-representing any single aspect.

Inspired by:
- Subtopic coverage in diversity-focused retrieval
- Maximal Marginal Relevance (MMR)
- Coverage-based summarization
"""
from typing import List, Dict, Any, Optional, Set
import copy
import math

from .base import ResultAggregator
from ..utils import Result


class CoverageAggregator(ResultAggregator):
    """
    Coverage-based aggregation that balances relevance and aspect coverage.
    
    The algorithm greedily selects documents that:
    1. Have high relevance scores across sub-queries
    2. Cover previously uncovered or under-covered query aspects
    
    This prevents the final ranking from being dominated by documents
    that are only relevant to one aspect of a complex query.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        coverage_weight: float = 0.5,
        top_k: Optional[int] = None,
        normalize_scores: bool = True,
    ):
        """
        Initialize the coverage aggregator.
        
        Args:
            config: Optional configuration object.
            coverage_weight: Weight for coverage vs relevance (0=pure relevance, 1=pure coverage).
            top_k: Maximum number of documents to return (None for all).
            normalize_scores: Whether to normalize scores before aggregation.
        """
        super().__init__(config)
        self.coverage_weight = coverage_weight
        self.top_k = top_k
        self.normalize_scores = normalize_scores

    def aggregate(
        self,
        sub_query_results: List[Result],
        weights: Optional[List[float]] = None,
        original_query: Optional[str] = None,
        qid: Optional[str] = None,
        **kwargs
    ) -> Result:
        """
        Aggregate using coverage-based selection.
        
        The algorithm:
        1. Compute relevance scores for each doc across all sub-queries
        2. Greedily select documents that maximize combined score:
           combined = (1 - coverage_weight) * relevance + coverage_weight * coverage_gain
        3. Update coverage state after each selection
        """
        if not sub_query_results:
            raise ValueError("sub_query_results cannot be empty")
        
        num_aspects = len(sub_query_results)
        if weights is None:
            weights = [1.0 / num_aspects] * num_aspects
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Build document-aspect relevance matrix
        doc_aspect_scores, doc_contents = self._build_score_matrix(
            sub_query_results, weights
        )
        
        # Greedy selection with coverage
        selected_docs = self._greedy_coverage_selection(
            doc_aspect_scores, num_aspects, weights
        )
        
        # Build result hits
        hits = []
        for rank, (docid, score) in enumerate(selected_docs, start=1):
            hits.append({
                'docid': docid,
                'score': score,
                'rank': rank,
                'content_dict': doc_contents.get(docid, {})
            })
        
        first_result = sub_query_results[0]
        return Result(
            qid=qid or first_result.qid,
            query=original_query or first_result.query,
            hits=hits
        )

    def _build_score_matrix(
        self,
        sub_query_results: List[Result],
        weights: List[float]
    ) -> tuple:
        """
        Build a matrix of document scores per aspect.
        
        Returns:
            doc_aspect_scores: Dict[docid, List[float]] - scores per aspect
            doc_contents: Dict[docid, Dict] - document contents
        """
        doc_aspect_scores: Dict[str, List[float]] = {}
        doc_contents: Dict[str, Dict[str, Any]] = {}
        
        for aspect_idx, result in enumerate(sub_query_results):
            # Compute score range for normalization
            if self.normalize_scores and result.hits:
                scores = [hit.get('score', 1.0 / (i + 1)) 
                         for i, hit in enumerate(result.hits)]
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score if max_score != min_score else 1.0
            else:
                min_score, score_range = 0.0, 1.0
            
            for rank, hit in enumerate(result.hits, start=1):
                docid = hit['docid']
                
                # Get or compute score
                raw_score = hit.get('score', 1.0 / rank)
                if self.normalize_scores:
                    normalized_score = (raw_score - min_score) / score_range
                else:
                    normalized_score = raw_score
                
                # Initialize if new document
                if docid not in doc_aspect_scores:
                    doc_aspect_scores[docid] = [0.0] * len(sub_query_results)
                
                doc_aspect_scores[docid][aspect_idx] = normalized_score
                
                # Store content
                if docid not in doc_contents:
                    doc_contents[docid] = hit.get('content_dict', {})
        
        return doc_aspect_scores, doc_contents

    def _greedy_coverage_selection(
        self,
        doc_aspect_scores: Dict[str, List[float]],
        num_aspects: int,
        weights: List[float]
    ) -> List[tuple]:
        """
        Greedy selection maximizing coverage + relevance.
        
        Returns:
            List of (docid, combined_score) tuples in selected order.
        """
        selected = []
        remaining_docs = set(doc_aspect_scores.keys())
        
        # Track coverage: how much each aspect is covered
        aspect_coverage = [0.0] * num_aspects
        
        # Determine how many docs to select
        max_docs = self.top_k if self.top_k else len(doc_aspect_scores)
        
        while remaining_docs and len(selected) < max_docs:
            best_doc = None
            best_score = float('-inf')
            
            for docid in remaining_docs:
                scores = doc_aspect_scores[docid]
                
                # Relevance component: weighted sum of scores
                relevance = sum(w * s for w, s in zip(weights, scores))
                
                # Coverage gain: how much new coverage does this doc add?
                coverage_gain = self._compute_coverage_gain(
                    scores, aspect_coverage, weights
                )
                
                # Combined score
                combined = ((1 - self.coverage_weight) * relevance + 
                           self.coverage_weight * coverage_gain)
                
                if combined > best_score:
                    best_score = combined
                    best_doc = docid
            
            if best_doc is None:
                break
            
            # Update coverage
            best_scores = doc_aspect_scores[best_doc]
            for i, score in enumerate(best_scores):
                aspect_coverage[i] = max(aspect_coverage[i], score)
            
            selected.append((best_doc, best_score))
            remaining_docs.remove(best_doc)
        
        return selected

    def _compute_coverage_gain(
        self,
        doc_scores: List[float],
        current_coverage: List[float],
        weights: List[float]
    ) -> float:
        """
        Compute the coverage gain from adding a document.
        
        Coverage gain is the weighted sum of improvements to under-covered aspects.
        """
        gain = 0.0
        for i, (doc_score, current, weight) in enumerate(
            zip(doc_scores, current_coverage, weights)
        ):
            # Gain is the improvement over current coverage
            improvement = max(0, doc_score - current)
            gain += weight * improvement
        return gain


class MMRAggregator(CoverageAggregator):
    """
    Maximal Marginal Relevance (MMR) aggregator.
    
    A variant of coverage aggregation that also considers document diversity
    by penalizing documents similar to already selected ones.
    
    MMR = lambda * relevance - (1-lambda) * max_similarity_to_selected
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        lambda_param: float = 0.7,
        top_k: Optional[int] = None,
        similarity_fn: Optional[callable] = None,
    ):
        """
        Args:
            config: Optional configuration.
            lambda_param: Balance between relevance and diversity (higher = more relevance).
            top_k: Maximum documents to return.
            similarity_fn: Function to compute similarity between documents.
        """
        super().__init__(
            config=config,
            coverage_weight=1 - lambda_param,
            top_k=top_k
        )
        self.lambda_param = lambda_param
        self._similarity_fn = similarity_fn

    def _compute_coverage_gain(
        self,
        doc_scores: List[float],
        current_coverage: List[float],
        weights: List[float]
    ) -> float:
        """
        For MMR, coverage gain represents diversity.
        Use parent's coverage gain as a proxy for diversity.
        """
        return super()._compute_coverage_gain(doc_scores, current_coverage, weights)
