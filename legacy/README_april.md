# APRIL
Accelerating Pairwise re-ranking with listwise prompts

---
## Requirements

- Environment & 
```
conda create -n april python=3.10
conda activate april
conda install -c conda-forge openjdk=21 maven -y
```

## TREC DL
```
# 2019
```
# reproduced


--- 

## Prepare runs 
The top-1000 run files used in this work can be found in [runs](runs/). 
We follow [Pyserini 2cr](https://castorini.github.io/pyserini/2cr/msmarco-v1-passage.html) for the reproduction.

```bash
# Retrieva with BM25 (+RM3) (K1=0.9, b=0.4) using Pyserini

python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl19-passage \
  --output runs/run.msmarco-passage.bm25-rm3-trec-dl-2019.txt \
  --bm25 --k1 0.9 --b 0.4 --rm3

python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl20-passage \
  --output runs/run.msmarco-passage.bm25-rm3-trec-dl-2020.txt \
  --bm25 --k1 0.9 --b 0.4 --rm3

python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl19-passage \
  --output runs/run.msmarco-passage.bm25-trec-dl-2019.txt \
  --bm25 --k1 0.9 --b 0.4

python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl20-passage \
  --output runs/run.msmarco-passage.bm25-trec-dl-2020.txt \
  --bm25 --k1 0.9 --b 0.4
```
