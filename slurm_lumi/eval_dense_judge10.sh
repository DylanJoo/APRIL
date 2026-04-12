cd $HOME/APRIL

DATASETS=(
"msmarco-passage@trec-dl-2019/judged"
"msmarco-passage@trec-dl-2020/judged"
"beir@dbpedia-entity/test"
"beir@nfcorpus/test"
"beir@scidocs"
"beir@trec-covid"
"beir@webis-touche2020/v2"
)

MODEL_DIR=meta-llama/Llama-3.3-70B-Instruct

# retrieval 
for retrieval in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for dataset in ${DATASETS[@]};do
    benchmark=$(echo $dataset | cut -d'@' -f1)
    subset=$(echo $dataset | cut -d'@' -f2)
    run_path=${HOME}/runs-and-qrels/runs/$benchmark/run.$benchmark.$retrieval.dataset.txt
    name=${subset%%/*}

    short_name=$(basename "$name" | cut -c1-3)
    j=$(python -m ir_measures $benchmark/$subset ${run_path/dataset/$name} Judged@10 | cut -f2) 
    echo "${retrieval} | - | ${name} | $j"
done
done

# reranking
for retrieval in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for rerank in point judge judge_expr setmaxheaptopk rankgpt;do
for dataset in ${DATASETS[@]};do
    benchmark=$(echo $dataset | cut -d'@' -f1)
    subset=$(echo $dataset | cut -d'@' -f2)
    run_path=${HOME}/APRIL/runs/${MODEL_DIR##*/}/run.$benchmark.$retrieval-rerank-$rerank.dataset.txt
    name=${subset%%/*}

    short_name=$(basename "$name" | cut -c1-3)
    j=$(python -m ir_measures $benchmark/$subset ${run_path/dataset/$name} Judged@10 | cut -f2) 
    echo "${retrieval} | ${rerank} | ${name} | $j"
done
done
done
