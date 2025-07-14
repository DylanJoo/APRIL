# msmarco-passage-v2. trec-dl 2021, 2022, 2023 
for year in 21 22 23;do
    python -m pyserini.search.lucene \
      --threads 16 --batch-size 128 \
      --index msmarco-v2-passage \
      --topics dl${year} \
      --output runs/run.msmarco-passage-v2.bm25.trec-dl-20${year}.txt \
      --bm25
done
