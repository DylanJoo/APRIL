# ENV
source ${HOME}/.bashrc
initconda
conda activate inference 

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

run_path=${HOME}/APRIL/runs/run.beir.bm25.dataset.txt

for id in {0..12};do
    dataset=${DATASETS[$id]}
    dataset=${dataset/beir./}
    short_name=$(basename "$dataset" | cut -c1-3)
    nDCG=$(python -m ir_measures ${QRELS[$id]} ${run_path/dataset/$dataset} nDCG@10 | cut -f2) 
    R=$(python -m ir_measures ${QRELS[$id]} ${run_path/dataset/$dataset} R@100 | cut -f2)
    echo "bm25 | ${short_name} | $nDCG | $R"
done
