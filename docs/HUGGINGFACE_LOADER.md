# HuggingFace Dataset Loader

This document describes how to use the HuggingFace dataset loader (`load_hf`) to load queries and corpus from HuggingFace datasets for use with the reranking pipeline.

## Overview

The `load_hf` function provides an alternative to the `load` function (which uses `ir_datasets`). It allows you to load queries and corpus directly from HuggingFace datasets.

## Function Signature

```python
def load_hf(
    queries_dataset_name: str,
    corpus_dataset_name: str,
    subset: str,
    query_split: str = 'test',
    corpus_split: str = 'train',
    qrels_split: Optional[str] = None,
    ignore_corpus: bool = False,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
```

## Parameters

- `queries_dataset_name` (str): HuggingFace dataset name for queries (e.g., 'DylanJHJ/nano-beir')
- `corpus_dataset_name` (str): HuggingFace dataset name for corpus (e.g., 'DylanJHJ/nano-beir-corpus')
- `subset` (str): Dataset subset/config name (e.g., 'nq', 'msmarco', 'fiqa')
- `query_split` (str, optional): Split name for queries dataset (default: 'test')
- `corpus_split` (str, optional): Split name for corpus dataset (default: 'train')
- `qrels_split` (str, optional): Split name for qrels (default: None, currently returns empty dict)
- `ignore_corpus` (bool, optional): If True, skip loading corpus and return None (default: False)

## Returns

A tuple of three dictionaries:
1. `corpus`: `Dict[str, Dict[str, str]]` - Maps document ID to `{"contents": text}`
2. `queries`: `Dict[str, str]` - Maps query ID to query text
3. `qrels`: `Dict[str, Dict[str, int]]` - Maps query ID to `{doc_id: relevance}` (currently empty)

## Expected Dataset Format

### Query Dataset
Expected fields:
- `query_id`: String or integer identifier for the query
- `query_texts`: The text of the query

### Corpus Dataset
Expected fields:
- `docid`: String or integer identifier for the document
- `title`: Document title (optional, can be empty)
- `text`: Document text content

## Usage Examples

### Example 1: Basic Loading

```python
from reranking import loader

# Load queries and corpus from HuggingFace datasets
corpus, queries, qrels = loader.load_hf(
    queries_dataset_name='DylanJHJ/nano-beir',
    corpus_dataset_name='DylanJHJ/nano-beir-corpus',
    subset='nq',
    query_split='test',
    corpus_split='train',
)

print(f"Loaded {len(queries)} queries")
print(f"Loaded {len(corpus)} documents")
```

### Example 2: Load Only Queries

```python
from reranking import loader

# Load only queries, skip corpus
corpus, queries, qrels = loader.load_hf(
    queries_dataset_name='DylanJHJ/nano-beir',
    corpus_dataset_name='DylanJHJ/nano-beir-corpus',
    subset='msmarco',
    ignore_corpus=True,
)

# corpus will be None
assert corpus is None
```

### Example 3: Integration with Reranking Pipeline

```python
from reranking import loader
from reranking.config_manager import ConfigManager
from reranking.wrapper import AutoLLMReranker

# Initialize reranker configuration
config = ConfigManager(
    rerank_mode='RankGPT',
    top_k=100,
    window_size=20,
    llm={'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct'}
).get_config()

# Load data from HuggingFace
corpus, queries, qrels = loader.load_hf(
    queries_dataset_name='DylanJHJ/nano-beir',
    corpus_dataset_name='DylanJHJ/nano-beir-corpus',
    subset='nq',
)

# Load initial run (BM25 baseline or other)
run = loader.load_run('path/to/run.txt')

# Initialize reranker
rankllm = AutoLLMReranker(
    config,
    system_message="You are RankLLM, an intelligent assistant..."
)

# Perform reranking
reranked_run = rankllm.rerank(
    run=run,
    queries=queries,
    corpus=corpus,
    query_batch_size=32,
)
```

### Example 4: Replacing ir_datasets with HuggingFace datasets

Before (using ir_datasets):
```python
corpus, queries, qrels = loader.load(
    ir_datasets_name='msmarco-passage/trec-dl-2019/judged',
    query_fields=None,
    doc_fields=None
)
```

After (using HuggingFace datasets):
```python
corpus, queries, qrels = loader.load_hf(
    queries_dataset_name='DylanJHJ/nano-beir',
    corpus_dataset_name='DylanJHJ/nano-beir-corpus',
    subset='nq',
)
```

## Differences from `load()` Function

| Feature | `load()` | `load_hf()` |
|---------|----------|-------------|
| Data source | ir_datasets | HuggingFace datasets |
| Query fields | Configurable via `query_fields` | Fixed: `query_id`, `query_texts` |
| Doc fields | Configurable via `doc_fields` | Fixed: `docid`, `title`, `text` |
| Qrels loading | Automatic from ir_datasets | Not yet implemented (returns empty dict) |
| Title handling | Depends on dataset | Automatically concatenated with text |

## Notes

1. **Qrels**: Currently, the function returns an empty dictionary for qrels. If you need qrels, you should load them separately or extend the function to handle them.

2. **Field Names**: The function expects specific field names in the HuggingFace datasets:
   - Queries: `query_id`, `query_texts`
   - Corpus: `docid`, `title`, `text`

3. **Title and Text**: The function automatically combines the title and text fields with a space separator. If only one is present, it uses that field.

4. **String Conversion**: All IDs (query_id, docid) are converted to strings for consistency.

## Error Handling

The function will raise exceptions if:
- The specified HuggingFace datasets cannot be found or accessed
- The datasets don't have the expected field names
- Network connection issues prevent downloading the datasets

Make sure you have:
- Internet connection to access HuggingFace datasets
- Required permissions to access private datasets (if applicable)
- The `datasets` library installed (`pip install datasets`)
