## Modularized LLM-Reranking Library: 
This modularized LLM-Reranking project provides a flexible framework for reranking search results using large language models (LLMs). 
It allows users to easily experiment and integrate different components from different neural reranking methods. 

[July 4, 2025] We consider `sorting algorithm`, `prompting template` and corresponding `ranking parser` with different `LLM backend`.

## Installation
- Environment
```bash
conda create -n rerank python=3.10
pip install uv
```
- Dependency Installation
```bash
git clone https://github.com/DylanJoo/APRIL.git
cd APRIL
uv pip install -e .
uv pip install vllm==0.11.1 ftfy ir_datasets ir_measures 
```
The final `requirements.txt` will be provided in the future release.

---

## Module Design Overview

The `autollmrerank` library follows a modular architecture where each component is responsible for a specific task in the reranking pipeline. The design enables flexible mixing-and-matching of different reranking strategies, prompt templates, LLM backends, and result parsing methods.

### Architecture Diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    AutoLLMReranker                      │
                    │           (Main Wrapper / Entry Point)                  │
                    └─────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
    ┌────────────────────────────────────────────────────────────────────────────────┐
    │                             AutoAssembler                                       │
    │        (Orchestrates reranking strategy: controls sorting algorithm)            │
    │                                                                                 │
    │   Strategies: SlidingWindow, PairAll, PairBubbleTopK, SetBubbleTopK, Point...   │
    └────────────────┬─────────────────────┬────────────────────┬────────────────────┘
                     │                     │                    │
                     ▼                     ▼                    ▼
    ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
    │   PromptBuilder    │   │    LLMProvider     │   │   ResultParser     │
    │                    │   │                    │   │                    │
    │ • Formats prompts  │   │ • vLLM backend     │   │ • Parses LLM text  │
    │ • Applies template │   │ • OpenAI API       │   │ • Extracts ranking │
    │ • Handles chat     │   │ • Request-based    │   │ • Computes scores  │
    └────────────────────┘   └────────────────────┘   └────────────────────┘
```

### Core Components

#### 1. AutoLLMReranker (Wrapper)
The main entry point for the reranking library. It:
- Loads prebuilt configurations via `from_prebuilt(method_name, model_name_or_path)`
- Manages the full reranking pipeline
- Converts input data to internal `Result` format and back to standard run format

#### 2. AutoAssembler (Input Assembler / Rerank Strategy)
Orchestrates the reranking process by implementing different sorting algorithms:

| Rerank Mode       | Algorithm            | Type      | Description                                |
|-------------------|----------------------|-----------|-------------------------------------------|
| `RankGPT`         | Sliding Window Bubble| Listwise  | Uses bubble sort with sliding windows      |
| `RankFirst`       | Sliding Window       | Listwise  | With distribution logprob output          |
| `PairAll`         | All-Pairs            | Pairwise  | Compares all pairs, aggregates scores     |
| `PairTopK`        | Bubble TopK          | Pairwise  | Bubbles top-k documents up using pairs    |
| `PairMaxHeapTopK` | MaxHeap TopK         | Pairwise  | Uses max-heap for efficient top-k         |
| `SetTopK`         | Bubble TopK          | Setwise   | Setwise comparisons with bubble sort      |
| `SetMaxHeapTopK`  | MaxHeap TopK         | Setwise   | Setwise comparisons with max-heap         |
| `Point`           | Independent Scoring  | Pointwise | Scores each document independently        |
| `Judge`           | Judge-based          | Judge     | LLM-as-a-judge scoring                    |

#### 3. PromptBuilder
Constructs prompts for the LLM using configurable templates:
- **Input**: Query and list of documents
- **Output**: Formatted prompts with chat template applied
- Supports system messages and various prompt formatters

#### 4. LLMProvider
Interfaces with different LLM backends for inference:
- **vLLM**: High-performance local inference with `AsyncLLMEngine`
- **OpenAI/Request**: API-based inference for hosted models
- Supports binary probability output (Yes/No) and distribution logprob output

#### 5. ResultParser
Parses LLM outputs and updates document rankings:
- **Response parsing**: Extracts permutation from text (e.g., "[1] > [3] > [2]")
- **Score parsing**: Handles absolute/partial scores from logprobs
- **Swap parsing**: Binary comparison results for pairwise methods

### Utility Classes
- **Result**: Encapsulates a query with its candidate documents and scores
- **ConfigManager**: Manages YAML configs with CLI argument overrides

---

## Usage

### Quick Start

```python
from autollmrerank import AutoLLMReranker
import ir_measures
from ir_measures import nDCG, RR  # Use @ operator for parameterized metrics

# Define your data
run = {
    "q1": {"d2": 0.95, "d1": 0.70, "d6": 0.25},
    "q2": {"d4": 0.88, "d3": 0.73, "d7": 0.20},
}

queries = {
    "q1": "What city is the capital of France?",
    "q2": "Who painted the Mona Lisa?",
}

corpus = {
    "d1": "Paris is the capital of France.",
    "d2": "London is the capital of the UK.",
    "d3": "Vincent van Gogh painted 'The Starry Night'.",
    "d4": "The painter of the Mona Lisa was Leonardo da Vinci.",
    "d6": "Berlin is the capital of Germany.",
    "d7": "Pablo Picasso painted 'Guernica'.",
}

qrel = {
    "q1": {"d1": 1},
    "q2": {"d4": 1},
}

# Initialize reranker with a prebuilt method
reranker = AutoLLMReranker.from_prebuilt('rankgpt', 'Qwen/Qwen2.5-7B-Instruct')

# Rerank documents
reranked_result = reranker.rerank(run=run, queries=queries, corpus=corpus)

# Evaluate results (ir_measures uses @ operator for parameterized metrics)
print(ir_measures.calc_aggregate([nDCG@10, RR@5], qrel, reranked_result))
```

### Using Different Reranking Methods

```python
# Listwise reranking with RankGPT (default sliding window bubble sort)
reranker = AutoLLMReranker.from_prebuilt('rankgpt', 'Qwen/Qwen2.5-7B-Instruct')

# Pointwise scoring
reranker = AutoLLMReranker.from_prebuilt('point', 'Qwen/Qwen2.5-7B-Instruct')

# Pairwise top-k with custom configuration
reranker = AutoLLMReranker.from_prebuilt(
    'pairtopk', 
    'Qwen/Qwen2.5-7B-Instruct',
    llm={'use_logits': True}
)

# Judge-based scoring
reranker = AutoLLMReranker.from_prebuilt('judge', 'Qwen/Qwen2.5-7B-Instruct')
```

### Command-Line Usage

You can run reranking directly from the command line:

```bash
python -m autollmrerank.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2020/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2020.txt \
    --llm.model_name_or_path=Qwen/Qwen2.5-7B-Instruct \
    --llm.max_model_len=8196 \
    --rerank_mode=RankGPT \
    --num_runs=1 \
    --window_size=20 \
    --step_size=10 \
    --dtype=float16
```

### Configuration Options

Key configuration parameters (can be set via YAML or CLI):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rerank_mode` | Reranking strategy to use | `RankGPT` |
| `llm.model_name_or_path` | HuggingFace model path or name | `Qwen/Qwen2.5-7B-Instruct` |
| `llm.backend` | LLM backend (`vllm`, `openai`, `request`) | `vllm` |
| `llm.use_logits` | Use logprob-based scoring | `false` |
| `llm.max_model_len` | Maximum model context length | `8196` |
| `llm.temperature` | Sampling temperature | `0.0` |
| `window_size` | Size of sliding window | `20` |
| `step_size` | Step size for sliding window | `10` |
| `top_k` | Number of documents to rerank | `100` |
| `num_runs` | Number of reranking passes | `1` |
| `use_alphabetical` | Use alphabetical identifiers (A, B, C...) | `false` |

---

## Project Structure

```
APRIL/
├── src/
│   └── autollmrerank/
│       ├── __init__.py           # Package initialization
│       ├── wrapper.py            # AutoLLMReranker main class
│       ├── config_manager.py     # Configuration loading and CLI parsing
│       ├── utils.py              # Result class and utilities
│       ├── loader.py             # Data loading utilities
│       ├── configs/              # Prebuilt YAML configurations
│       │   ├── default.yaml
│       │   ├── rankgpt.yaml
│       │   ├── point.yaml
│       │   ├── pairtopk.yaml
│       │   └── ...
│       ├── input_assembler/      # Reranking strategies
│       │   ├── base.py           # RerankStrategy abstract base
│       │   ├── auto.py           # AutoAssembler factory
│       │   ├── list_bubble.py    # SlidingWindow strategies
│       │   ├── pair_all.py       # PairAll strategy
│       │   ├── pair_bubble_topk.py
│       │   ├── set_bubble_topk.py
│       │   ├── point.py          # Pointwise scoring
│       │   └── judge.py          # LLM-as-judge
│       ├── prompt_builder/       # Prompt construction
│       │   ├── base.py           # PromptBuilder class
│       │   └── formatter/        # Prompt templates
│       ├── llm_provider/         # LLM inference backends
│       │   ├── vllm.py           # vLLM async engine
│       │   ├── vllm_dev.py       # vLLM development version
│       │   └── request.py        # API request-based
│       └── result_parser/        # Output parsing
│           └── base.py           # ResultParser class
├── example/                      # Example scripts
│   ├── README.md
│   ├── run_trec-dl-2020.sh
│   └── ...
├── unittest/                     # Unit tests
└── pyproject.toml
```

