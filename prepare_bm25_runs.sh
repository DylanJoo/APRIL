# reproduced
# python -m pyserini.search.lucene \
#   --threads 16 --batch-size 128 \
#   --index msmarco-v1-passage \
#   --topics dl19-passage \
#   --output run.msmarco-v1-passage.bm25-rm3-default.dl19.txt \
#   --bm25 --k1 0.9 --b 0.4 --rm3

# evaluation
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage \
  runs/run.msmarco-v1-passage.bm25-rm3-default.dl19.txt
python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl19-passage \
  runs/run.msmarco-v1-passage.bm25-rm3-default.dl19.txt

