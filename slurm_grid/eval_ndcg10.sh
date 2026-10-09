# ENV
source ${HOME}/.bashrc
initconda
conda activate autollmreranker 

DATASETS=(
"msmarco-passage@trec-dl-2019/judged"
"msmarco-passage@trec-dl-2020/judged"
"beir@dbpedia-entity/test"
"beir@nfcorpus/test"
"beir@scidocs"
"beir@trec-covid"
"beir@webis-touche2020/v2"
)

# retrieval 
for retrieval in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for dataset in ${DATASETS[@]};do
    benchmark=$(echo $dataset | cut -d'@' -f1)
    subset=$(echo $dataset | cut -d'@' -f2)
    run_path=${HOME}/runs-and-qrels/runs/$benchmark/run.$benchmark.$retrieval.dataset.txt
    name=${subset%%/*}

    short_name=$(basename "$name" | cut -c1-3)
    nDCG=$(python -m ir_measures $benchmark/$subset ${run_path/dataset/$name} nDCG@10 | cut -f2) 
    echo "${retrieval} | - | ${name} | $nDCG"
done
done

# reranking
for retrieval in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for rerank in point umbrela judge judge_expr setmaxheaptopk umbrela_setmaxheaptopk rankgpt umbrela_rankgpt;do
for dataset in ${DATASETS[@]};do
    benchmark=$(echo $dataset | cut -d'@' -f1)
    subset=$(echo $dataset | cut -d'@' -f2)
    run_path=${HOME}/runs-and-qrels/runs/$benchmark/run.$benchmark.$retrieval-rerank-$rerank.dataset.txt
    name=${subset%%/*}

    short_name=$(basename "$name" | cut -c1-3)
    nDCG=$(python -m ir_measures $benchmark/$subset ${run_path/dataset/$name} nDCG@10 | cut -f2) 
    echo "${retrieval} | ${rerank} | ${name} | $nDCG"
done
done
done

# supervised reranking
for retrieval in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for rerank in rankfirst rankzephyr;do
for dataset in ${DATASETS[@]};do
    benchmark=$(echo $dataset | cut -d'@' -f1)
    subset=$(echo $dataset | cut -d'@' -f2)
    run_path=${HOME}/runs-and-qrels/runs/$benchmark/run.$benchmark.$retrieval-rerank-$rerank.dataset.txt
    name=${subset%%/*}

    short_name=$(basename "$name" | cut -c1-3)
    nDCG=$(python -m ir_measures $benchmark/$subset ${run_path/dataset/$name} nDCG@10 | cut -f2) 
    echo "${retrieval} | ${rerank} | ${name} | $nDCG"
done
done
done
