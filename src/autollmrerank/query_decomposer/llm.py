"""
LLM-based query decomposition.

Uses a language model to decompose complex queries into simpler sub-queries.
This is particularly useful for:
- Multi-faceted queries with multiple aspects
- Complex questions that can be broken into simpler ones
- Queries that benefit from different perspectives or reformulations
"""
from typing import List, Dict, Any, Optional
import re

from .base import QueryDecomposer, DecomposedQuery


DEFAULT_DECOMPOSITION_PROMPT = """You are a query decomposition assistant. Given a complex search query, break it down into simpler, more specific sub-queries that together cover all aspects of the original query.

Original Query: {query}

Instructions:
1. Identify the main intent(s) and aspects of the query
2. Create 2-4 focused sub-queries that together capture the full meaning
3. Each sub-query should be independently searchable
4. Return sub-queries as a numbered list

Sub-queries:"""


class LLMDecomposer(QueryDecomposer):
    """
    Uses an LLM to decompose queries into sub-queries.
    
    The LLM is prompted to identify different aspects or intents
    within a complex query and generate focused sub-queries.
    """

    def __init__(
        self,
        llm_provider: Any,
        config: Optional[Any] = None,
        prompt_template: Optional[str] = None,
        max_sub_queries: int = 4,
        min_sub_queries: int = 1,
    ):
        """
        Initialize the LLM decomposer.
        
        Args:
            llm_provider: The LLM provider to use for generation.
            config: Optional configuration object.
            prompt_template: Custom prompt template for decomposition.
                            Must contain {query} placeholder.
            max_sub_queries: Maximum number of sub-queries to generate.
            min_sub_queries: Minimum number of sub-queries to return.
        """
        super().__init__(config)
        self._llm = llm_provider
        self._prompt_template = prompt_template or DEFAULT_DECOMPOSITION_PROMPT
        self._max_sub_queries = max_sub_queries
        self._min_sub_queries = min_sub_queries

    def decompose(self, query: str, **kwargs) -> DecomposedQuery:
        """
        Decompose a query using the LLM.
        
        Args:
            query: The query to decompose.
            **kwargs: Additional arguments (e.g., temperature, context).
            
        Returns:
            DecomposedQuery with generated sub-queries.
        """
        prompt = self._prompt_template.format(query=query)
        
        # Generate using LLM
        outputs = self._llm.generate([prompt])
        response = outputs[0] if outputs else ""
        
        # Parse sub-queries from response
        sub_queries = self._parse_sub_queries(response, query)
        
        # Calculate weights (can be uniform or based on other heuristics)
        weights = self._calculate_weights(sub_queries, query)
        
        return DecomposedQuery(
            original_query=query,
            sub_queries=sub_queries,
            weights=weights,
            metadata={
                "method": "llm",
                "raw_response": response,
                "prompt": prompt
            }
        )

    def decompose_batch(
        self, 
        queries: List[str], 
        **kwargs
    ) -> List[DecomposedQuery]:
        """
        Decompose multiple queries in batch for efficiency.
        
        Args:
            queries: List of queries to decompose.
            **kwargs: Additional arguments.
            
        Returns:
            List of DecomposedQuery objects.
        """
        prompts = [self._prompt_template.format(query=q) for q in queries]
        outputs = self._llm.generate(prompts)
        
        results = []
        for query, response in zip(queries, outputs):
            sub_queries = self._parse_sub_queries(response, query)
            weights = self._calculate_weights(sub_queries, query)
            results.append(DecomposedQuery(
                original_query=query,
                sub_queries=sub_queries,
                weights=weights,
                metadata={
                    "method": "llm",
                    "raw_response": response
                }
            ))
        return results

    def _parse_sub_queries(self, response: str, original_query: str) -> List[str]:
        """
        Parse sub-queries from LLM response.
        
        Handles various formats:
        - Numbered lists (1. query, 2. query)
        - Bulleted lists (- query, * query)
        - Line-separated queries
        """
        # Try to extract numbered list items
        numbered_pattern = r'(?:\d+[\.\)]\s*)(.+?)(?=\n\d+[\.\)]|\n*$)'
        matches = re.findall(numbered_pattern, response, re.MULTILINE)
        
        if matches:
            sub_queries = [m.strip() for m in matches if m.strip()]
        else:
            # Try bullet points
            bullet_pattern = r'(?:[-*•]\s*)(.+?)(?=\n[-*•]|\n*$)'
            matches = re.findall(bullet_pattern, response, re.MULTILINE)
            if matches:
                sub_queries = [m.strip() for m in matches if m.strip()]
            else:
                # Fall back to line-by-line
                lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
                sub_queries = lines if lines else [original_query]
        
        # Enforce min/max constraints
        if len(sub_queries) < self._min_sub_queries:
            sub_queries = [original_query]
        elif len(sub_queries) > self._max_sub_queries:
            sub_queries = sub_queries[:self._max_sub_queries]
            
        return sub_queries

    def _calculate_weights(
        self, 
        sub_queries: List[str], 
        original_query: str
    ) -> List[float]:
        """
        Calculate weights for sub-queries.
        
        Default implementation uses equal weights.
        Can be overridden for more sophisticated weighting schemes.
        """
        n = len(sub_queries)
        return [1.0 / n] * n
