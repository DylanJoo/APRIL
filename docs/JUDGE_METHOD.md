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
  - Computes score as P(rating=5) / sum(all rating probabilities)
  - Similar to binary_probs which uses yes / (yes + no)
  - Returns score in range [0, 1] representing confidence in highest rating

#### Implementation Details:
```python
# In set_classification():
self.rating_tokens = {
    i: [tokenizer.encode(s)[0] for s in rating_strings if s.strip() == str(i)]
    for i in range(6)
}

# In _iterate_over_output():
rating_probs = [exp(max_logprob_for_rating_i) for i in 0..5]
score = rating_probs[5] / sum(rating_probs)  # P(rating=5) / total
```

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
3. **Probability-Based Scoring**: Returns P(rating=5) / sum(all ratings) similar to binary_probs
4. **Flexible**: Can be used with or without logit trick
5. **Consistent API**: Follows same pattern as other reranking methods

## Differences from Point Method

| Aspect | Point | Judge |
|--------|-------|-------|
| Question | "Is this relevant?" | "How relevant is this?" |
| Response | Yes/No | 0-5 rating |
| Logit Tokens | Yes/No tokens | 0-5 tokens |
| Score Computation | yes/(yes+no) | rating5/sum(all ratings) |
| Score Range | [0, 1] probability | [0, 1] probability |
| Use Case | Binary relevance | Graded relevance |

## Integration with Logit Trick

The logit trick provides several advantages:
- **Efficiency**: Only needs first token probabilities (no full generation)
- **Calibration**: Uses model's uncertainty in rating predictions
- **Normalized Scores**: P(rating=5) / sum(all ratings) provides scores in [0, 1]
- **Similar to Binary**: Follows same pattern as binary_probs (yes/no)

### Scoring Method

The current implementation uses:
```python
score = P(rating=5) / sum(P(rating=0), P(rating=1), ..., P(rating=5))
```

This is analogous to the binary approach where `score = yes / (yes + no)`, treating rating 5 as the "positive" class.

### Potential Extensions

Future scoring variations could include:
1. **Threshold-based**: `P(rating≥3) / sum(all ratings)` - probability of being at least moderately relevant
2. **Multi-threshold**: `P(rating≥4) / sum(all ratings)` - probability of being highly relevant
3. **Weighted combination**: `(3×P(rating=3) + 4×P(rating=4) + 5×P(rating=5)) / sum(all ratings)` - weighted by rating levels
4. **Expected value**: `Σ(i × P(rating=i)) / 5` - normalized expected rating

These extensions would allow for different sensitivity levels and use cases depending on the retrieval task requirements.
