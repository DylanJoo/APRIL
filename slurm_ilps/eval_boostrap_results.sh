# ENV
source ${HOME}/.bashrc
initconda
conda activate inference 
cd ${HOME}/APRIL

DATASETS=(
"beir.arguana"
"beir.climate-fever"
"beir.dbpedia-entity"
"beir.fever"
"beir.fiqa"
"beir.hotpotqa"
"beir.nfcorpus"
"beir.nq"
"beir.quora"
"beir.scidocs"
"beir.scifact"
"beir.trec-covid"
"beir.webis-touche2020"
)

QRELS=(
"beir/arguana"
"beir/climate-fever"
"beir/dbpedia-entity/test"
"beir/fever/test"
"beir/fiqa/test"
"beir/hotpotqa/test"
"beir/nfcorpus/test"
"beir/nq"
"beir/quora/test"
"beir/scidocs"
"beir/scifact/test"
"beir/trec-covid"
"beir/webis-touche2020/v2"
)


for boopstrap_id in {1..10};do
for method in bm25-null judge judge_expr point rankfirst rankgpt rankzephyr setmaxheap;do
for id in {0..10};do # 11 and 12 only needs to be done once as smaller than 50
    dataset=${DATASETS[$id]}
    dataset=${dataset/beir./}
    short_name=$(basename "$dataset" | cut -c1-3)

    run_path=bbier/run_${boopstrap_id}/${method}.beir-${dataset}.txt
    # nDCG=$(python -m ir_measures ${QRELS[$id]} $run_path nDCG@10 | cut -f2) 
    # echo "${method} | ${short_name} | $id | $nDCG "
  
    nDCG=$(python src/autollmrerank/eval_bootstrap.py --irds_tag ${QRELS[$id]} --path $run_path)
    echo "${method} | ${short_name} | $boopstrap_id | $nDCG " 
done
done
done
