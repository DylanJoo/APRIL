# qrel-analysis

A pipeline for evaluating how well LLM-judge-derived relevance judgments (qrels) align with human judgments, across multiple thresholding strategies and retrieval systems.

---

## Overview

This pipeline:
1. Takes LLM judge scores (e.g., reranking scores) and converts them into binary qrels using various thresholding strategies
2. Evaluates a retrieval run against those derived qrels using nDCG@10
3. Aggregates results across datasets and judge systems into pivot CSVs

---

## Pipeline

```
Human QRels + LLM Judge Runs + Retrieval Runs
            ↓
    eval_autoqrels.py          ← convert judge scores to qrels, evaluate
            ↓
    Per-strategy nDCG@10 results (JSONL)
            ↓
    collect_results.py         ← aggregate across judges/datasets
            ↓
    Pivot CSVs (one per retriever)
```

---

## Scripts

### 1. `eval_autoqrels.py` — Core evaluation

Converts LLM judge scores to binary qrels using a thresholding strategy, then evaluates a retrieval run against those qrels.

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--dataset_name` | Yes | — | Dataset name (e.g., `beir/nfcorpus/test`) |
| `--loader_type` | No | `irds` | Loader module type |
| `--judge_run` | Yes | — | Path to LLM judge JSONL file (used as qrel source) |
| `--evaluate_run` | No | same as `judge_run` | Path to retrieval run JSONL to evaluate |
| `--strategies` | No | — | One or more strategies (see below); repeatable |
| `--threshold` | No | `0.5` | Score cutoff for `thresholding` strategy |
| `--rank_cutoff` | No | `10` | Top-k docs treated as relevant for `rank` strategy |
| `--gap_k` | No | `1` | k-th largest score gap for `largest_gap` strategy |
| `--quantile_cutoff` | No | `0.75` | Quantile threshold for `quantile` strategy |
| `--min_relevance` | No | `1` | Min human relevance grade for oracle strategies |
| `--exp` | No | — | Optional experiment tag added to output records |

**Thresholding strategies:**

| Strategy | Oracle? | Description |
|---|---|---|
| `direct` | No | Round LLM scores to nearest integer |
| `thresholding` | No | Binary threshold at `--threshold` (default 0.5) |
| `rank` | No | Top-k documents are relevant (k = `--rank_cutoff`) |
| `largest_gap` | No | Threshold at the k-th largest score gap |
| `quantile` | No | Score >= q-th percentile is relevant |
| `optimal_per_topic` | Yes | Per-topic threshold maximizing F1 vs human qrels |
| `optimal_global` | Yes | Single global threshold maximizing macro-avg F1 |
| `all` | — | Apply all strategies at once |

**Input format** (both `--judge_run` and `--evaluate_run`): JSONL, one record per line:
```json
{"qid": "query_id", "docid": "doc_id", "score": 3.14}
```

**Output format:** JSONL to stdout, one record per strategy:
```json
{"dataset": "beir/nfcorpus/test", "exp": "judge:eval", "strategy": "direct", "nDCG@10": 0.5559}
```

**Example usage:**
```bash
# Evaluate using a single strategy
python eval_autoqrels.py \
  --dataset_name beir/nfcorpus/test \
  --judge_run /path/to/judge_run.jsonl \
  --evaluate_run /path/to/bm25_run.jsonl \
  --strategies rank

# Multiple strategies
python eval_autoqrels.py \
  --dataset_name beir/nfcorpus/test \
  --judge_run /path/to/judge_run.jsonl \
  --evaluate_run /path/to/bm25_run.jsonl \
  --strategies direct --strategies rank --strategies largest_gap

# All strategies with experiment tag, save to file
python eval_autoqrels.py \
  --dataset_name beir/nfcorpus/test \
  --judge_run /path/to/judge_run.jsonl \
  --evaluate_run /path/to/bm25_run.jsonl \
  --strategies all \
  --exp "my_experiment" \
  > results/my_judge/raaj-nfcorpus.jsonl
```

---

### 2. `collect_results.py` — Aggregate results

Scans result directories for JSONL files (`*/raaj*.jsonl`), parses them, and produces pivot-table CSVs grouped by retrieval system.

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--results_dir` | Yes | Base directory containing per-judge subdirectories |
| `--output_dir` | Yes | Directory to write one CSV per retrieval system |

**Expected input directory structure:**
```
results_dir/
├── bm25-rerank-judge/
│   ├── raaj-nfcorpus.jsonl
│   ├── raaj-trec-covid.jsonl
│   └── ...
├── colbert-small-rerank-judge/
│   └── ...
└── splade-v3-rerank-judge/
    └── ...
```

**Output:** One CSV per retrieval system (e.g., `bm25.csv`, `colbert-small.csv`), pipe-delimited pivot tables comparing nDCG@10 across judge systems and strategies.

**Example usage:**
```bash
python collect_results.py \
  --results_dir ./results \
  --output_dir ./new
```

---

### 3. `get_beir_stats.py` — Dataset statistics

Prints a tab-separated statistics table for BEIR and MS MARCO datasets. No arguments needed.

**Datasets covered:** msmarco-passage/trec-dl-2019, trec-dl-2020, and 13 BEIR datasets (arguana, climate-fever, dbpedia-entity, fever, fiqa, hotpotqa, nfcorpus, nq, quora, scidocs, scifact, trec-covid, webis-touche2020).

**Statistics reported:** num queries, corpus size, total judgments, avg query/doc length, avg positives/negatives per query, judgment level range.

```bash
python get_beir_stats.py
```

---

## Results Directory

Current results are in `results/`, organized by `{retriever}-rerank-{judge}/`, e.g.:
- `bm25-rerank-judge/`
- `colbert-small-rerank-judge/`
- `splade-v3-rerank-judge/`
- `nomicai-modernbert-embed-rerank-judge/`
- `qwen3-embed-600m-rerank-judge/`

Each subdirectory contains `raaj-{dataset}.jsonl` files.

---

## Dependencies

- `autollmrerank` — internal module for dataset loading (`loader_dev.irds`)
- `ir_measures` — IR evaluation metrics
- `pandas` — result aggregation
