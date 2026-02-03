# Design Review & Query Decomposition Integration Proposal

## Current Architecture Overview

The APRIL library follows a **Strategy Pattern** combined with a **Factory Pattern** for modularity. Here's the current architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AutoLLMReranker (wrapper.py)                 │
│  - Entry point for reranking                                        │
│  - Orchestrates the entire pipeline                                 │
│  - Handles batching at the query level                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AutoAssembler (Factory)                          │
│  - Maps rerank_mode → Strategy implementation                       │
│  - Creates concrete strategy with injected dependencies             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │   RerankStrategy    │         │   Other Strategies  │
        │   (list_bubble.py)  │         │   (pair_all, etc.)  │
        └─────────────────────┘         └─────────────────────┘
                    │
        ┌───────────┼───────────┬───────────────┐
        ▼           ▼           ▼               ▼
   PromptBuilder  LLMProvider  ResultParser  ConfigManager
```

## Feedback on Current Design

### Strengths ✅

1. **Good Separation of Concerns**: The four core modules (InputAssembler/Strategy, PromptBuilder, LLMProvider, ResultParser) each have clear responsibilities.

2. **Factory Pattern**: `AutoAssembler` and `AutoPromptFormatter` provide clean instantiation based on configuration, making it easy to add new strategies.

3. **Strategy Pattern**: `RerankStrategy` provides a good abstraction for different reranking algorithms (sliding window, pairwise, pointwise, setwise).

4. **Configuration-Driven**: YAML configs with CLI overrides provide flexibility without code changes.

5. **Dependency Injection**: Strategies receive their dependencies (prompt_builder, llm_provider, result_parser) via constructor injection, enabling testability.

### Areas for Improvement 🔧

#### 1. **Tight Coupling in `AutoLLMReranker.__init__`**

The wrapper directly imports and instantiates the LLM provider based on backend type:

```python
if config.llm.backend == 'vllm':
    from .llm_provider.vllm import LLM
if (config.llm.backend == 'openai') or (config.llm.backend == 'request'):
    from .llm_provider.request import LLM
```

**Suggestion**: Create an `AutoLLMProvider` factory class (similar to `AutoAssembler`) that handles this mapping:

```python
# llm_provider/auto.py
class AutoLLMProvider:
    _provider_map = {
        'vllm': 'autollmrerank.llm_provider.vllm.LLM',
        'openai': 'autollmrerank.llm_provider.request.LLM',
        'request': 'autollmrerank.llm_provider.request.LLM',
        'vllm_dev': 'autollmrerank.llm_provider.vllm_dev.LLM',
    }
    
    @classmethod
    def from_config(cls, config):
        # Lazy import and instantiate
        ...
```

#### 2. **Missing Abstract Base for LLMProvider**

Unlike `RerankStrategy` and `BaseFormatter`, the LLM providers don't have a formal base class/interface defining the expected contract. This makes it unclear what methods a new LLM provider must implement.

**Suggestion**: Create `llm_provider/base.py` with an abstract base class:

```python
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompts, **kwargs):
        """Generate responses for given prompts."""
        pass
    
    @abstractmethod
    def set_classification(self, id_strings):
        """Configure for classification mode."""
        pass
```

#### 3. **Inconsistent Naming Convention**

- `InputAssembler` vs `RerankStrategy` (the folder is `input_assembler` but the base class is `RerankStrategy`)
- Some methods use `run_pass` while others don't support it

**Suggestion**: Standardize naming. Consider renaming `input_assembler` to `rerank_strategy` or vice versa.

#### 4. **Result Object Could Be More Robust**

The `Result` class is simple but could benefit from:
- Type hints for `hits` structure
- Validation methods
- Immutability options (frozen dataclass)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Hit:
    docid: str
    score: float
    content_dict: Dict[str, str]
    rank: Optional[int] = None

@dataclass
class Result:
    qid: str
    query: str
    hits: List[Hit] = field(default_factory=list)
    ranking_exec_summary: Optional[Dict] = None
```

#### 5. **Hardcoded Values in Strategies**

Some strategies have hardcoded values that should be configurable:

```python
# list_bubble.py line 34
hit['score'] = float(1 / rank)  # Reciprocal rank - could be configurable
```

---

## Query Decomposition Integration Design

### Conceptual Overview

Query decomposition breaks a complex query into simpler sub-queries, performs reranking for each, and aggregates results. This fits naturally as a **pre-processing stage** before the existing reranking pipeline.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          AutoLLMReranker                                   │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    QueryDecomposer (NEW)                                   │
│  - Decomposes complex queries into sub-queries                             │
│  - Uses LLMProvider for decomposition                                      │
│  - Optional: can be bypassed for simple queries                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌─────────────┐                 ┌─────────────┐
            │ Sub-query 1 │       ...       │ Sub-query N │
            └─────────────┘                 └─────────────┘
                    │                               │
                    ▼                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                       AutoAssembler (existing)                             │
│  - Runs reranking for each sub-query                                       │
└────────────────────────────────────────────────────────────────────────────┘
                    │                               │
                    ▼                               ▼
            ┌─────────────┐                 ┌─────────────┐
            │  Results 1  │       ...       │  Results N  │
            └─────────────┘                 └─────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    ResultAggregator (NEW)                                  │
│  - Combines results from all sub-queries                                   │
│  - Strategies: fusion, voting, weighted sum, etc.                          │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            Final Ranked Results
```

### Proposed Implementation

#### 1. New Module: `query_decomposer/`

```
src/autollmrerank/
├── query_decomposer/
│   ├── __init__.py
│   ├── base.py           # Abstract base class
│   ├── auto.py           # Factory
│   ├── llm_decomposer.py # LLM-based decomposition
│   └── prompt_templates/ # Decomposition prompts
│       └── default.txt
```

**Base Class:**

```python
# query_decomposer/base.py
from abc import ABC, abstractmethod
from typing import List, Tuple
from ..utils import Result

class QueryDecomposer(ABC):
    """Base class for query decomposition strategies."""
    
    @abstractmethod
    def decompose(self, query: str) -> List[str]:
        """
        Decompose a query into sub-queries.
        
        Args:
            query: The original query string
            
        Returns:
            List of sub-query strings (including original if appropriate)
        """
        pass
    
    @abstractmethod
    def should_decompose(self, query: str) -> bool:
        """
        Determine if a query should be decomposed.
        
        Args:
            query: The query to evaluate
            
        Returns:
            True if decomposition is beneficial
        """
        pass


class NoOpDecomposer(QueryDecomposer):
    """Pass-through decomposer that returns the original query."""
    
    def decompose(self, query: str) -> List[str]:
        return [query]
    
    def should_decompose(self, query: str) -> bool:
        return False
```

**LLM-based Decomposer:**

```python
# query_decomposer/llm_decomposer.py
from typing import List
from .base import QueryDecomposer

class LLMQueryDecomposer(QueryDecomposer):
    """Uses an LLM to decompose complex queries into sub-queries."""
    
    def __init__(self, llm_provider, config):
        self.llm = llm_provider
        self.config = config
        self.decompose_prompt = self._load_prompt_template()
        self.max_subqueries = config.get('max_subqueries', 3)
        self.include_original = config.get('include_original', True)
    
    def decompose(self, query: str) -> List[str]:
        if not self.should_decompose(query):
            return [query]
        
        prompt = self.decompose_prompt.format(query=query)
        response = self.llm.generate([prompt])[0]
        sub_queries = self._parse_subqueries(response)
        
        if self.include_original:
            sub_queries = [query] + sub_queries
        
        return sub_queries[:self.max_subqueries]
    
    def should_decompose(self, query: str) -> bool:
        # Simple heuristics - can be made smarter
        word_count = len(query.split())
        has_conjunction = any(w in query.lower() for w in ['and', 'or', 'but', 'also'])
        has_multiple_aspects = '?' in query or has_conjunction
        
        return word_count > 10 or has_multiple_aspects
    
    def _parse_subqueries(self, response: str) -> List[str]:
        """Parse LLM response into list of sub-queries."""
        # Parse numbered list format: "1. query1\n2. query2\n..."
        lines = response.strip().split('\n')
        sub_queries = []
        for line in lines:
            # Remove numbering and clean
            clean = line.strip().lstrip('0123456789.-) ').strip()
            if clean:
                sub_queries.append(clean)
        return sub_queries
    
    def _load_prompt_template(self) -> str:
        return """Break down the following search query into simpler, focused sub-queries.
Each sub-query should capture a different aspect or requirement of the original query.
Return 2-3 sub-queries, one per line, numbered.

Original query: {query}

Sub-queries:"""
```

#### 2. New Module: `result_aggregator/`

```
src/autollmrerank/
├── result_aggregator/
│   ├── __init__.py
│   ├── base.py           # Abstract base class
│   ├── auto.py           # Factory
│   ├── fusion.py         # Reciprocal Rank Fusion
│   ├── voting.py         # Voting-based aggregation
│   └── weighted.py       # Weighted combination
```

**Base Class:**

```python
# result_aggregator/base.py
from abc import ABC, abstractmethod
from typing import List, Dict
from ..utils import Result

class ResultAggregator(ABC):
    """Base class for aggregating results from multiple sub-queries."""
    
    @abstractmethod
    def aggregate(
        self, 
        sub_results: List[List[Result]], 
        sub_queries: List[str],
        original_query: str
    ) -> List[Result]:
        """
        Aggregate results from multiple sub-query reranking runs.
        
        Args:
            sub_results: List of result lists, one per sub-query
            sub_queries: The sub-queries used
            original_query: The original query
            
        Returns:
            Aggregated results
        """
        pass
```

**Reciprocal Rank Fusion:**

```python
# result_aggregator/fusion.py
from typing import List, Dict
from collections import defaultdict
import copy
from .base import ResultAggregator
from ..utils import Result

class ReciprocalRankFusion(ResultAggregator):
    """
    Implements Reciprocal Rank Fusion (RRF) for combining ranked lists.
    RRF score = sum(1 / (k + rank)) across all lists
    """
    
    def __init__(self, k: int = 60, weights: List[float] = None):
        self.k = k
        self.weights = weights
    
    def aggregate(
        self, 
        sub_results: List[List[Result]], 
        sub_queries: List[str],
        original_query: str
    ) -> List[Result]:
        
        # Determine weights
        weights = self.weights or [1.0] * len(sub_results)
        
        # Group by qid (same qid across sub-queries)
        aggregated = []
        
        # Assuming sub_results[i] contains results for sub_query[i]
        # and each has the same structure (same qids)
        qid_to_results = defaultdict(lambda: defaultdict(float))
        qid_to_hits = {}
        
        for weight, results in zip(weights, sub_results):
            for result in results:
                qid = result.qid
                if qid not in qid_to_hits:
                    qid_to_hits[qid] = {}
                
                for rank, hit in enumerate(result.hits, start=1):
                    docid = hit['docid']
                    rrf_score = weight * (1.0 / (self.k + rank))
                    qid_to_results[qid][docid] += rrf_score
                    
                    # Store hit data (take from first occurrence)
                    if docid not in qid_to_hits[qid]:
                        qid_to_hits[qid][docid] = copy.deepcopy(hit)
        
        # Build aggregated results
        for qid, doc_scores in qid_to_results.items():
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            hits = []
            for rank, (docid, score) in enumerate(sorted_docs, start=1):
                hit = qid_to_hits[qid][docid]
                hit['score'] = score
                hit['rank'] = rank
                hits.append(hit)
            
            aggregated.append(Result(
                qid=qid,
                query=original_query,
                hits=hits
            ))
        
        return aggregated
```

#### 3. Updated `AutoLLMReranker` with Decomposition Support

```python
# wrapper.py (modified)
class AutoLLMReranker:
    
    def __init__(self, config, **kwargs) -> None:
        self.config = config
        
        # Existing initialization...
        prompt_builder = PromptBuilder(config=config)
        agent = self._create_llm_provider(config)
        result_parser = ResultParser(use_alpha=config.use_alphabetical)
        
        self.assembler = AutoAssembler.from_config(
            config, 
            prompt_builder=prompt_builder,
            llm_provider=agent,
            result_parser=result_parser,
        )
        
        # NEW: Initialize decomposition components if enabled
        if getattr(config, 'enable_decomposition', False):
            from .query_decomposer import AutoDecomposer
            from .result_aggregator import AutoAggregator
            
            self.decomposer = AutoDecomposer.from_config(config, llm_provider=agent)
            self.aggregator = AutoAggregator.from_config(config)
        else:
            self.decomposer = None
            self.aggregator = None
    
    def rerank(self, run, queries, corpus, query_batch_size=32):
        init_results = self.convert_run_to_result(run, queries, corpus)
        
        if self.decomposer:
            return self._rerank_with_decomposition(init_results, query_batch_size)
        else:
            return self._rerank_standard(init_results, query_batch_size)
    
    def _rerank_with_decomposition(self, init_results, query_batch_size):
        """Rerank with query decomposition and result aggregation."""
        all_reranked = []
        
        for result in tqdm(init_results, desc="Processing queries with decomposition"):
            # Decompose query
            sub_queries = self.decomposer.decompose(result.query)
            
            if len(sub_queries) == 1:
                # No decomposition needed
                reranked = self._run_single_rerank([result], query_batch_size)
                all_reranked.extend(reranked)
            else:
                # Run reranking for each sub-query
                sub_results = []
                for sub_query in sub_queries:
                    # Create modified result with sub-query
                    sub_result = copy.deepcopy(result)
                    sub_result.query = sub_query
                    
                    reranked = self._run_single_rerank([sub_result], query_batch_size)
                    sub_results.append(reranked)
                
                # Aggregate results
                aggregated = self.aggregator.aggregate(
                    sub_results, 
                    sub_queries, 
                    result.query
                )
                all_reranked.extend(aggregated)
        
        return self._convert_results_to_run(all_reranked)
    
    def _rerank_standard(self, init_results, query_batch_size):
        """Standard reranking without decomposition."""
        reranked_results = self._run_single_rerank(init_results, query_batch_size)
        return self._convert_results_to_run(reranked_results)
    
    def _run_single_rerank(self, results, query_batch_size):
        """Run a single reranking pass."""
        reranked_results = []
        for batch_results in batch_iterator(results, size=query_batch_size):
            batch_reranked = self.assembler.run(
                init_results=batch_results, 
                rank_start=0,
                rank_end=min(self.config.rank_end, self.config.top_k),
                batch_size=query_batch_size,
                num_runs=self.config.num_runs,
            )
            reranked_results.extend(batch_reranked)
        
        for r in reranked_results:
            r.sort_by(field='score')
        
        return reranked_results
```

#### 4. Configuration Extensions

```yaml
# configs/decomposition.yaml
decomposition:
  enabled: true
  method: llm  # Options: llm, rule_based, hybrid
  max_subqueries: 3
  include_original: true
  min_query_length: 10  # Only decompose queries with >= N words
  
aggregation:
  method: rrf  # Options: rrf, voting, weighted, max
  rrf_k: 60
  weights: null  # Optional: [1.0, 0.8, 0.6] for sub-query weights
```

### Alternative Design: Decomposition as a Strategy

Instead of pre/post processing stages, decomposition could be implemented as a `RerankStrategy`:

```python
# input_assembler/decomposed.py
class DecomposedRerank(RerankStrategy):
    """
    A meta-strategy that decomposes queries and delegates to inner strategy.
    """
    
    def __init__(self, config, inner_strategy, decomposer, aggregator, **kwargs):
        super().__init__(config, **kwargs)
        self.inner_strategy = inner_strategy
        self.decomposer = decomposer
        self.aggregator = aggregator
    
    def run(self, init_results, rank_start, rank_end, **kwargs):
        all_aggregated = []
        
        for result in init_results:
            sub_queries = self.decomposer.decompose(result.query)
            
            sub_results = []
            for sub_query in sub_queries:
                sub_result = copy.deepcopy(result)
                sub_result.query = sub_query
                reranked = self.inner_strategy.run([sub_result], rank_start, rank_end, **kwargs)
                sub_results.append(reranked)
            
            aggregated = self.aggregator.aggregate(sub_results, sub_queries, result.query)
            all_aggregated.extend(aggregated)
        
        return all_aggregated
```

This approach uses the **Decorator Pattern** - wrapping an existing strategy with decomposition capabilities.

### Recommendations

1. **Start Simple**: Implement the pre/post processing approach first (cleaner separation).

2. **Make Decomposition Optional**: Use `enable_decomposition` config flag.

3. **Share LLM Provider**: Reuse the same LLM for both decomposition and reranking.

4. **Add Caching**: Cache decomposition results for repeated queries.

5. **Implement Multiple Aggregators**: Start with RRF, then add voting, weighted, etc.

6. **Add Metrics**: Track decomposition stats (# sub-queries, aggregation method used).

### File Structure After Implementation

```
src/autollmrerank/
├── __init__.py
├── config_manager.py
├── utils.py
├── wrapper.py
├── loader.py
├── configs/
│   ├── default.yaml
│   ├── decomposition.yaml      # NEW
│   └── ...
├── input_assembler/
│   ├── __init__.py
│   ├── base.py
│   ├── auto.py
│   ├── decomposed.py           # NEW (optional decorator approach)
│   └── ...
├── prompt_builder/
│   └── ...
├── llm_provider/
│   ├── __init__.py
│   ├── base.py                 # NEW (abstract base)
│   ├── auto.py                 # NEW (factory)
│   └── ...
├── result_parser/
│   └── ...
├── query_decomposer/           # NEW MODULE
│   ├── __init__.py
│   ├── base.py
│   ├── auto.py
│   ├── llm_decomposer.py
│   ├── rule_based.py
│   └── prompts/
│       └── decompose.txt
└── result_aggregator/          # NEW MODULE
    ├── __init__.py
    ├── base.py
    ├── auto.py
    ├── fusion.py
    ├── voting.py
    └── weighted.py
```

---

## Summary

### Current Design Strengths
- Good modular structure with Strategy and Factory patterns
- Configuration-driven with CLI overrides
- Dependency injection for testability

### Suggested Improvements
1. Add `AutoLLMProvider` factory to reduce coupling
2. Create abstract base class for LLM providers
3. Standardize naming conventions
4. Enhance `Result` class with type hints

### Query Decomposition Design
- Add two new modules: `query_decomposer/` and `result_aggregator/`
- Implement as pre/post processing stages in `AutoLLMReranker`
- Alternative: Decorator pattern wrapping existing strategies
- Share LLM provider between decomposition and reranking
- Start with RRF aggregation, expand to other methods
