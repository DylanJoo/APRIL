# reproduced
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl19-passage \
  --output runs/run.msmarco-v1-passage.bm25-rm3-trec-dl-2019.txt \
  --bm25 --k1 0.9 --b 0.4 --rm3

python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl20-passage \
  --output runs/run.msmarco-v1-passage.bm25-rm3-trec-dl-2020.txt \
  --bm25 --k1 0.9 --b 0.4 --rm3

python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl19-passage \
  --output runs/run.msmarco-v1-passage.bm25-trec-dl-2019.txt \
  --bm25 --k1 0.9 --b 0.4

python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl20-passage \
  --output runs/run.msmarco-v1-passage.bm25-trec-dl-2020.txt \
  --bm25 --k1 0.9 --b 0.4

