## Modularized LLM-Reranking Library: 
This modularized LLM-Reranking project provides a flexible framework for reranking search results using large language models (LLMs). 
It allows users to easily experiment and integrate different components from different neural reranking methods. 

[July 4, 2025] We consider `sorting algorithm`, `prompting template` and corresponding `rankign parser` with different `LLM backend`.

## Installation
- Environment
```
conda create -n rerank python=3.10
pip install uv
```
- Dependency Installation
```
git clone https://github.com/DylanJoo/APRIL.git
cd APRIL
uv pip install -e .
uv pip install vllm==0.11.1 ftfy ir_datasets ir_measures 
```
The final `requirements.txt` will be provided in the future release.

## Structure
```
APRIL/
├── unittest/                     # Unit tests
│   ├── test_query_decomposer.py
│   ├── test_result_aggregator.py
│   └── test_multi_query_assembler.py
├── src/
│   └── autollmrerank/
│       ├── config_manager.py
│       ├── utils.py
│       ├── wrapper.py            # Base AutoLLMReranker
│       ├── input_assembler/      # Reranking strategies
│       │   ├── list_bubble.py    # SlidingWindow (RankGPT/RankZephyr)
│       │   ├── pair_*.py         # Pairwise strategies
│       │   ├── set_*.py          # Setwise strategies
│       │   └── multi_query.py    # Multi-query reranking with aggregation
│       ├── prompt_builder/       # Prompt formatting
│       ├── llm_provider/         # LLM backends
│       ├── result_parser/        # Output parsing
│       ├── query_decomposer/     # Data structures for sub-queries
│       │   └── base.py           # DecomposedQuery dataclass
│       └── result_aggregator/    # Post-ranking aggregation
│           ├── base.py           # RRF, PassThrough
│           └── coverage.py       # Coverage-based, MMR
└── example/
    └── demo_multi_query_reranking.py
```

## Core Architecture

### Wrapper/Main Functions
- **AutoLLMReranker**: Base wrapper class integrating the four core modules

### Four Core Modules
- **InputAssembler**: Defines reranking strategy (listwise, pairwise, setwise, multi-query)
    * Input: query and results
    * Output: list of query-documents pairs

- **PromptBuilder**: Formats prompts for the LLM
    * Input: query and documents 
    * Output: text prompts for LLM

- **LLMProvider**: Calls the LLM backend
    * Input: text prompts
    * Output: text outputs or list of numbers

- **ResultParser**: Parses LLM output into rankings
    * Input: text outputs or list of numbers
    * Output: Result object with sorted results

### Multi-Query Reranking
For complex queries that can be decomposed into multiple aspects, use `MultiQueryAssembler`:

1. **Sub-queries are generated externally** (not in this framework - use a separate query decomposition service)
2. **Attach sub-queries to a Result** using the helper function
3. **MultiQueryAssembler** reranks with each sub-query and aggregates results

```python
from autollmrerank.input_assembler import attach_sub_queries, MultiQueryAssembler
from autollmrerank.result_aggregator import RRFAggregator

# Attach sub-queries to your result (sub-queries generated externally)
attach_sub_queries(
    result,
    sub_queries=['What is machine learning?', 'ML applications in industry'],
    weights=[0.6, 0.4]  # optional weights
)

# Create multi-query assembler
assembler = MultiQueryAssembler(
    config=config,
    prompt_builder=prompt_builder,
    llm_provider=llm,
    result_parser=result_parser,
    base_strategy=base_strategy,  # e.g., SlidingWindow
    aggregator=RRFAggregator(k=60),
)

# Rerank - will use each sub-query and aggregate results
reranked = assembler.run(init_results=[result], rank_start=0, rank_end=100)
```

### Result Aggregation
Aggregates results from multiple sub-query rerankings into a final ranking.

- **PassThroughAggregator**: Returns first result unchanged (baseline)
- **RRFAggregator**: Reciprocal Rank Fusion
- **CoverageAggregator**: Coverage-based selection balancing relevance and aspect coverage
- **MMRAggregator**: Maximal Marginal Relevance for diversity

```python
from autollmrerank.result_aggregator import RRFAggregator, CoverageAggregator

# Reciprocal Rank Fusion
aggregator = RRFAggregator(k=60)
final = aggregator.aggregate(sub_query_results, weights=[0.5, 0.5])

# Coverage-based aggregation
aggregator = CoverageAggregator(coverage_weight=0.5, top_k=100)
final = aggregator.aggregate(sub_query_results, weights=[0.5, 0.5])
```

### Utility Classes
- **Result**: Represents retrieval/ranking results
- **DecomposedQuery**: Data structure for sub-queries and weights (for external use)

