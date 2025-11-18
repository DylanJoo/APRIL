# HuggingFace Dataset Loader

This document describes how to use the HuggingFace dataset loader with the APRIL reranking pipeline.

## Overview

The `load_hf()` function in `reranking.loader` enables loading queries and corpus from HuggingFace datasets, providing an alternative to the `ir-datasets` library.

## Installation

Make sure you have the HuggingFace `datasets` library installed:

```bash
pip install datasets
```

## Usage

### Basic Usage

```python
from reranking import loader

# Load queries and corpus from HuggingFace datasets
corpus, queries, qrels = loader.load_hf(
    dataset_name_queries='DylanJHJ/nano-beir',
    dataset_name_corpus='DylanJHJ/nano-beir-corpus',
    subset='nfcorpus'
)
```

### Function Signature

```python
def load_hf(
    dataset_name_queries: str,
    dataset_name_corpus: str,
    subset: str,
    query_split: str = 'test',
    corpus_split: str = 'corpus',
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
    ignore_corpus: bool = False,
    qrels_split: Optional[str] = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]
```

### Parameters

- `dataset_name_queries` (str): HuggingFace dataset name for queries (e.g., 'DylanJHJ/nano-beir')
- `dataset_name_corpus` (str): HuggingFace dataset name for corpus (e.g., 'DylanJHJ/nano-beir-corpus')
- `subset` (str): Dataset subset/configuration name (e.g., 'nfcorpus', 'scifact', 'fiqa')
- `query_split` (str, optional): Split name for queries (default: 'test')
- `corpus_split` (str, optional): Split name for corpus (default: 'corpus')
- `query_fields` (list, optional): List of query fields to concatenate (default: ['query_texts'])
- `doc_fields` (list, optional): List of document fields to concatenate (default: ['title', 'text'])
- `ignore_corpus` (bool, optional): If True, skip loading corpus (default: False)
- `qrels_split` (str, optional): Split name for qrels (default: None). If None, qrels are not loaded.

### Returns

A tuple of `(corpus, queries, qrels)`:
- `corpus`: Dict mapping `doc_id` to dict with 'contents' key
- `queries`: Dict mapping `query_id` to query text string
- `qrels`: Dict mapping `query_id` to dict of `doc_id` to relevance score

## Expected Dataset Format

### Query Dataset

The query dataset should have the following fields:
- `query_id` (str): Unique query identifier
- `query_texts` (str): Query text

### Corpus Dataset

The corpus dataset should have the following fields:
- `docid` (str): Unique document identifier
- `title` (str, optional): Document title
- `text` (str): Document content

### Qrels Dataset (Optional)

If available, the qrels dataset should have:
- `query_id` or `qid` (str): Query identifier
- `doc_id` or `docid` (str): Document identifier
- `relevance` or `score` (int): Relevance score

## Examples

### Example 1: Load Specific Subset

```python
from reranking import loader

# Load the 'nfcorpus' subset
corpus, queries, qrels = loader.load_hf(
    dataset_name_queries='DylanJHJ/nano-beir',
    dataset_name_corpus='DylanJHJ/nano-beir-corpus',
    subset='nfcorpus',
    query_split='test',
    corpus_split='corpus'
)

print(f"Loaded {len(queries)} queries")
print(f"Loaded {len(corpus)} documents")
```

### Example 2: Custom Field Selection

```python
# Use only document titles
corpus, queries, qrels = loader.load_hf(
    dataset_name_queries='DylanJHJ/nano-beir',
    dataset_name_corpus='DylanJHJ/nano-beir-corpus',
    subset='scifact',
    doc_fields=['title']  # Only use title field
)
```

### Example 3: Integration with Reranking Pipeline

```python
from reranking import loader
from reranking.config_manager import ConfigManager
from reranking.wrapper import ModularReranker

# Load data from HuggingFace
corpus, queries, qrels = loader.load_hf(
    dataset_name_queries='DylanJHJ/nano-beir',
    dataset_name_corpus='DylanJHJ/nano-beir-corpus',
    subset='nfcorpus'
)

# Load initial BM25 run
run = loader.load_run('runs/run.bm25.nfcorpus.txt')

# Initialize reranker
config = ConfigManager(
    rerank_mode='RankGPT',
    top_k=100,
    llm={'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct'}
).get_config()

rankllm = ModularReranker(config)

# Rerank using HuggingFace data
reranked_run = rankllm.rerank(
    run=run,
    queries=queries,
    corpus=corpus,
    query_batch_size=64
)
```

### Example 4: Load Queries Only

```python
# Skip corpus loading for query-only operations
corpus, queries, qrels = loader.load_hf(
    dataset_name_queries='DylanJHJ/nano-beir',
    dataset_name_corpus='DylanJHJ/nano-beir-corpus',
    subset='nfcorpus',
    ignore_corpus=True  # Don't load corpus
)

# corpus will be None
print(queries)  # Only queries are loaded
```

## Comparison with ir-datasets Loader

The `load_hf()` function maintains API compatibility with the existing `load()` function:

| Feature | `load()` (ir-datasets) | `load_hf()` (HuggingFace) |
|---------|------------------------|---------------------------|
| Data source | ir-datasets library | HuggingFace Hub |
| Return format | Same | Same |
| Query format | Dict[str, str] | Dict[str, str] |
| Corpus format | Dict[str, Dict[str, str]] | Dict[str, Dict[str, str]] |
| Qrels format | Dict[str, Dict[str, int]] | Dict[str, Dict[str, int]] |
| Field selection | Supported | Supported |
| Ignore corpus | Supported | Supported |

Both loaders return data in the same format, making them interchangeable in the reranking pipeline.

## Notes

- The HuggingFace `datasets` library must be installed to use this function
- Internet access is required to download datasets from the HuggingFace Hub
- Downloaded datasets are cached locally for subsequent use
- Field names are case-sensitive and must match the dataset schema

## See Also

- [Example script](../example_hf_loader.py) - Complete usage examples
- [Test script](../test_hf_loader.py) - Unit tests and validation
