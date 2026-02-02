# Judge Pointwise Reranking Method

## Overview
This implementation adds a new pointwise reranking method called "Judge" that prompts LLMs to rate query-document pairs on a 0-5 relevance scale. The method supports both standard text generation and the "logit trick" for more efficient inference.

## Components Added

### 1. JudgeFormatter (`src/autollmrerank/prompt_builder/formatter/judge.py`)
- **Purpose**: Creates prompts for the Judge reranking method
- **Prompt Structure**:
  - Passage and query information
  - Rating scale explanation (0-5)
  - Clear instructions to output only a number
- **Key Method**: `postfix()` includes the rating scale descriptions

### 2. Judge Input Assembler (`src/autollmrerank/input_assembler/judge.py`)
- **Purpose**: Orchestrates the reranking process
- **Process**:
  1. Creates prompts for all query-document pairs
  2. Batches prompts for efficient processing
  3. Calls LLM with `rating_probs=True` for logit-based scoring
  4. Aggregates scores and updates results
- **Similar to**: Point (pointwise) assembler, but uses rating scores instead of binary

### 3. LLM Provider Updates
Updated both `vllm.py` and `request.py` to support rating-based logit extraction:

#### Added Features:
- **`rating_tokens`**: Dictionary mapping ratings (0-5) to their token IDs
- **`rating_probs` parameter**: New flag for generate() method
- **Logit Processing**:
  - Extracts logit probabilities for tokens "0" through "5"
  - Normalizes probabilities across ratings
  - Computes expected rating as weighted average
  - Returns continuous score in range [0, 5]

### 4. Configuration File (`src/autollmrerank/configs/judge.yaml`)
- Based on the Point configuration
- Sets `rerank_mode: Judge`
- Includes appropriate system message for judge role
- Configures `use_logits: true` for logit trick

### 5. Unit Test (`unittest/judge.py`)
- Example usage of Judge method
- Tests on TREC-DL 2019/2020 datasets
- Evaluates with nDCG@10 metric
- Follows same pattern as other reranking methods

### 6. Auto-Registration
Updated registration in:
- `src/autollmrerank/prompt_builder/formatter/auto.py`: Added JudgeFormatter
- `src/autollmrerank/input_assembler/auto.py`: Added Judge assembler

## Usage

### Basic Usage:
```python
from autollmrerank.config_manager import ConfigManager
from autollmrerank.wrapper import ModularReranker

config = ConfigManager(
    rerank_mode='Judge',
    top_k=100,
    llm={'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct', 'use_logits': True}
).get_config()

rankllm = ModularReranker(
    config, 
    system_message="You are JudgeLLM, an intelligent assistant that can judge the relevance of a passage to a query"
)

reranked_results = rankllm.rerank(run=run, queries=queries, corpus=corpus)
```

## Key Features

1. **Rating Scale**: Uses 0-5 scale for more nuanced relevance judgments
2. **Logit Trick**: Efficiently extracts ratings from token probabilities
3. **Continuous Scores**: Returns expected rating (weighted average) for better ranking
4. **Flexible**: Can be used with or without logit trick
5. **Consistent API**: Follows same pattern as other reranking methods

## Differences from Point Method

| Aspect | Point | Judge |
|--------|-------|-------|
| Question | "Is this relevant?" | "How relevant is this?" |
| Response | Yes/No | 0-5 rating |
| Logit Tokens | Yes/No tokens | 0-5 tokens |
| Score Range | [0, 1] probability | [0, 5] expected rating |
| Use Case | Binary relevance | Graded relevance |
