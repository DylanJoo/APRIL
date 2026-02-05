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
│   └── test_result_aggregator.py
├── src/
│   └── autollmrerank/
│       ├── config_manager.py
│       ├── utils.py
│       ├── wrapper.py            # Base AutoLLMReranker
│       ├── wrapper_extended.py   # Extended reranker with pre/post-ranking
│       ├── input_assembler/      # Reranking strategies
│       ├── prompt_builder/       # Prompt formatting
│       ├── llm_provider/         # LLM backends
│       ├── result_parser/        # Output parsing
│       ├── query_decomposer/     # Pre-ranking module
│       │   ├── base.py           # Base classes
│       │   └── llm.py            # LLM-based decomposition
│       └── result_aggregator/    # Post-ranking module
│           ├── base.py           # Base classes + RRF
│           └── coverage.py       # Coverage-based aggregation
└── example/
    └── demo_extended_reranker.py
```

## Core Architecture

### Wrapper/Main Functions
- **AutoLLMReranker**: Base wrapper class integrating the four core modules
- **ExtendedAutoLLMReranker**: Extended wrapper with pre-ranking and post-ranking support

### Four Core Modules
- **InputAssembler**: Defines reranking strategy (listwise, pairwise, setwise)
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

### Pre-Ranking Module (Query Decomposition)
Decomposes complex queries into simpler sub-queries before reranking.

- **PassThroughDecomposer**: No decomposition (baseline)
- **LLMDecomposer**: Uses LLM to break queries into aspects/sub-queries

```python
from autollmrerank.query_decomposer import PassThroughDecomposer, LLMDecomposer

# Passthrough (no decomposition)
decomposer = PassThroughDecomposer()
result = decomposer.decompose("What is ML and its applications?")
# result.sub_queries = ["What is ML and its applications?"]

# LLM-based decomposition (requires LLM provider)
decomposer = LLMDecomposer(llm_provider=agent, max_sub_queries=4)
result = decomposer.decompose("What is ML and its applications?")
# result.sub_queries = ["What is machine learning?", "ML applications in industry"]
```

### Post-Ranking Module (Result Aggregation)
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

### Extended Pipeline
The full pipeline with pre-ranking and post-ranking:

```python
from autollmrerank.wrapper_extended import ExtendedAutoLLMReranker

# Create extended reranker
reranker = ExtendedAutoLLMReranker.from_prebuilt(
    method_name="RankGPT",
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    decomposer="llm",           # or "passthrough"
    aggregator="coverage",       # or "rrf", "mmr", "passthrough"
    decomposer_config={"max_sub_queries": 4},
    aggregator_config={"coverage_weight": 0.5},
)

# Rerank with decomposition and aggregation
reranked_run = reranker.rerank(
    run=initial_run,
    queries=queries,
    corpus=corpus,
    use_decomposition=True
)
```

### Utility Classes
- **Result**: Represents retrieval/ranking results
- **DecomposedQuery**: Holds original query, sub-queries, and weights
- **AggregatedResult**: Holds aggregated results with metadata

