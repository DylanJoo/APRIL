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

## Supported Reranking Modes

### Listwise Methods
- `RankGPT`: Sliding window approach for listwise ranking
- `RankGPT+`: Enhanced sliding window with improvements
- `RankZephyr`: Zephyr-based listwise ranking
- `RankFirst`: Rank-first sliding window approach

### Pairwise Methods
- `PairAll`: All pairwise comparisons
- `PairTopK`: Bubble-sort based pairwise top-k
- `PairMaxHeapTopK`: Max-heap based pairwise top-k

### Setwise Methods
- `SetTopK`: Bubble-sort based setwise top-k
- `SetMaxHeapTopK`: Max-heap based setwise top-k

### Pointwise Methods
- `Point`: Basic pointwise Yes/No relevance scoring

### LLM-as-a-Judge Methods (NEW)
- `Judge`: Flexible pointwise judgment with multiple scoring modes
- `JudgeFewShot`: Few-shot reference-based judgment
- `JudgeEnsemble`: Ensemble of multiple scoring methods

#### Judge Scoring Modes
- `binary`: Simple Yes/No relevance judgment
- `rating`: Rate on a scale (default 0-5)
- `rubric_binary`: Use rubric context but output binary

#### Judge Scoring Computation Methods
- `binary_probs`: P(Yes) / (P(Yes) + P(No))
- `peak_likelihood`: logP(target_rating) or exp(logP(target_rating))
- `normalized_softmax`: Softmax over selected rating tokens
- `expected_rating`: Weighted sum of P(rating) * rating
- `rubric_binary`: Binary with rubric context

#### Example Usage for Judge Mode
```python
from autollmrerank.config_manager import ConfigManager
from autollmrerank.wrapper import AutoLLMReranker

# Configure Judge mode with expected rating
config = ConfigManager(
    path='src/autollmrerank/configs/judge.yaml',
    rerank_mode='Judge',
    scoring_mode='rating',
    scoring_computation='expected_rating',
    rating_scale=5,
    llm={'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct'}
).get_config()

reranker = AutoLLMReranker(config)
reranked_run = reranker.rerank(run=run, queries=queries, corpus=corpus)
```

## Structure [TODO: update to the beta version]
```
APRIL/ # the proposed new method using `autollmrerank`.
├── unittest/li_textlist.py
├── src/
│   └── autollmrerank/
│       ├── __init__.py
│       ├── config_manager.py
│       ├── utils.py
│       ├── prompt_builder/
│       │   ├── base.py
│       │   └── _rank_gpt.py
│       ├── llm_provider/
│       │   ├── base.py
│       │   └── _rank_gpt.py
│       ├── result_parser/
│       │   ├── base.py
│       │   └── _rank_gpt.py
│       └── tests/
│           └── test_model.py
└── examples/
    ├── run_trec-dl-2020.sh
    └── log.vllm/
```

#### Wrapper/main functions
- ModularReranker 
    * A wrapper class that defines the reranking types for class factory, which can integrates all the following 4 components.

#### Four modules 
- InputAssebler (rename? scheduler? handler? ...)
    * Input: query and results
    * Output: list of query-documents pairs

- PromptBuilder
    * Input: query and documents 
    * Output: text prompts for LLM

- LLMProvider
    * Input: text prompts
    * output: text outputs or list of numbers

- ResultParser
    * Input: text outputs or list of numbers
    * Output: Result object with sorted results

#### Utililty functions/classes
- Result: the class of retrieval/ranking results.
- PromptMode: the class of reranking mode, including the prompt, llm calling and parsing


