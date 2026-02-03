"""
Judge: Input assembler for LLM-as-a-Judge scoring.

Supports multiple scoring computation modes:
- binary_probs: Yes/(Yes+No) probability (existing pointwise)
- peak_likelihood: logP of the target rating
- normalized_softmax: Softmax over rating tokens
- expected_rating: Sum of P(rating) * rating values
- rubric_binary: Binary judgment with rubric context
"""
import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result, batch_iterator
from .base import RerankStrategy


class Judge(RerankStrategy):
    """
    Judge-based pointwise scoring for relevance assessment.
    
    Scoring modes:
    - 'binary_probs': P(Yes) / (P(Yes) + P(No))
    - 'peak_likelihood': logP(target_rating) or exp(logP(target_rating))
    - 'normalized_softmax': softmax over selected rating tokens
    - 'expected_rating': weighted sum of P(rating) * rating
    - 'rubric_binary': binary with rubric context
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Judge-specific settings from config
        self.scoring_mode = getattr(self.config, 'scoring_mode', 'binary')
        self.scoring_computation = getattr(self.config, 'scoring_computation', 'binary_probs')
        self.rating_scale = getattr(self.config, 'rating_scale', 5)
        self.rating_threshold = getattr(self.config, 'rating_threshold', 3)
        self.target_ratings = getattr(self.config, 'target_ratings', None)  # e.g., [5] or [4, 5] or [3, 4, 5]
        self.use_log_scale = getattr(self.config, 'use_log_scale', False)

    def run(
        self,
        init_results: List[Result],
        rank_start: int = 0,
        rank_end: int = None,
        batch_size: Optional[int] = 32,
        references: Optional[List[Dict]] = None,
        **kwargs
    ) -> List[Result]:
        """
        Run the judge scoring over all results.
        
        Args:
            init_results: List of Result objects with hits to score
            rank_start: Start index for scoring
            rank_end: End index for scoring
            batch_size: Batch size for LLM generation
            references: Optional few-shot reference examples
        """
        results = [copy.deepcopy(result) for result in init_results]
        all_scores = {}

        for index, result in enumerate(results):
            # Placeholder for scores
            result.hits = [hit for hit in result.hits[:rank_end]]
            all_scores[result.qid] = [0.0 for _ in result.hits]

            # Create prompts for all documents
            prompts = self._prompt_builder.create_prompt(
                result, 
                rank_start=0, 
                rank_end=rank_end,
                references=references
            )

            # Generate scores in batches
            scores = []
            for batch_prompts in tqdm(
                batch_iterator(prompts, batch_size),
                desc=f"Judge scoring with batch size {batch_size}",
            ):
                batch_scores = self._generate_scores(batch_prompts)
                scores.extend(batch_scores)

            # Aggregate scores
            for i, score in enumerate(scores):
                all_scores[result.qid][i] = score

        # Update results with scores
        reranked_results = self._result_parser.parse(
            [all_scores[result.qid] for result in results],
            init_results
        )
        return reranked_results

    def _generate_scores(self, prompts: List[str]) -> List[float]:
        """
        Generate scores based on the scoring computation mode.
        
        Returns:
            List of scores for each prompt
        """
        if self.scoring_computation == 'binary_probs':
            # Yes/(Yes+No) probability
            return self._llm.generate(prompts, binary_probs=True)
        
        elif self.scoring_computation == 'peak_likelihood':
            # logP of target rating(s)
            return self._llm.generate(prompts, rating_logp=True, 
                                       target_ratings=self.target_ratings or [self.rating_scale],
                                       rating_scale=self.rating_scale,
                                       use_log_scale=self.use_log_scale)
        
        elif self.scoring_computation == 'normalized_softmax':
            # Softmax over target ratings
            return self._llm.generate(prompts, rating_softmax=True,
                                       target_ratings=self.target_ratings or [self.rating_scale],
                                       rating_scale=self.rating_scale)
        
        elif self.scoring_computation == 'expected_rating':
            # Expected value: sum of P(rating) * rating
            return self._llm.generate(prompts, expected_rating=True,
                                       rating_scale=self.rating_scale)
        
        elif self.scoring_computation == 'rubric_binary':
            # Binary with rubric context
            return self._llm.generate(prompts, binary_probs=True)
        
        else:
            # Fallback to binary
            return self._llm.generate(prompts, binary_probs=True)

    def run_pass(self, **kwargs: Any):
        raise NotImplementedError("Judge does not support `run_pass`. Use run instead.")


class JudgeFewShot(Judge):
    """
    Extended Judge with enhanced few-shot reference support.
    
    Reference modes:
    - 'one_shot': Single reference example
    - 'two_shot': Two reference examples
    - 'pseudo': Model-generated pseudo-references
    
    Reference structures:
    - 'positive_only': Only positive examples
    - 'with_negative': Include negative examples
    - 'tight': Examples near decision boundary
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_references = getattr(self.config, 'num_references', 1)
        self.reference_structure = getattr(self.config, 'reference_structure', 'positive_only')

    def run(
        self,
        init_results: List[Result],
        rank_start: int = 0,
        rank_end: int = None,
        batch_size: Optional[int] = 32,
        references: Optional[List[Dict]] = None,
        **kwargs
    ) -> List[Result]:
        """
        Run few-shot judge scoring.
        
        If references are not provided, can generate pseudo-references.
        """
        if references is None and self.reference_structure == 'pseudo':
            references = self._generate_pseudo_references(init_results)
        
        return super().run(
            init_results=init_results,
            rank_start=rank_start,
            rank_end=rank_end,
            batch_size=batch_size,
            references=references,
            **kwargs
        )

    def _generate_pseudo_references(self, results: List[Result]) -> List[Dict]:
        """
        Generate pseudo-references from the current result set.
        
        Uses top/bottom documents as positive/negative examples.
        """
        pseudo_refs = []
        
        for result in results[:1]:  # Use first query for references
            if len(result.hits) < 2:
                continue
            
            query = result.query
            
            # Top document as positive example
            top_hit = result.hits[0]
            pseudo_refs.append({
                'query': query,
                'passage': top_hit.get('content_dict', {}).get('text', ''),
                'judgment': 'Yes' if self.scoring_mode == 'binary' else str(self.rating_scale)
            })
            
            # Bottom document as negative example (if needed)
            if self.reference_structure == 'with_negative' and len(result.hits) > 1:
                bottom_hit = result.hits[-1]
                pseudo_refs.append({
                    'query': query,
                    'passage': bottom_hit.get('content_dict', {}).get('text', ''),
                    'judgment': 'No' if self.scoring_mode == 'binary' else '0'
                })
        
        return pseudo_refs[:self.num_references]


class JudgeEnsemble(Judge):
    """
    Judge with ensemble scoring from multiple rating perspectives.
    
    Combines scores from different scoring computations for robust ranking.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_methods = getattr(self.config, 'ensemble_methods', 
                                        ['binary_probs', 'expected_rating'])
        self.ensemble_weights = getattr(self.config, 'ensemble_weights', None)

    def _generate_scores(self, prompts: List[str]) -> List[float]:
        """
        Generate ensemble scores from multiple computation methods.
        """
        if self.ensemble_weights is None:
            weights = [1.0 / len(self.ensemble_methods)] * len(self.ensemble_methods)
        else:
            weights = self.ensemble_weights
        
        ensemble_scores = [0.0] * len(prompts)
        
        for method, weight in zip(self.ensemble_methods, weights):
            # Temporarily switch scoring computation
            original_computation = self.scoring_computation
            self.scoring_computation = method
            
            scores = super()._generate_scores(prompts)
            
            # Restore original computation
            self.scoring_computation = original_computation
            
            # Weighted aggregation
            for i, score in enumerate(scores):
                ensemble_scores[i] += weight * score
        
        return ensemble_scores
