# Configuraitons — same-retrieval setting
# Judge = retrieval A reranked by LLM judge B (A-rerank-B).
# Candidates = only systems built on A (A alone + A-rerank-* for every reranker).

DATASETS=(
"msmarco-passage@trec-dl-2019/judged"
"msmarco-passage@trec-dl-2020/judged"
"beir@dbpedia-entity/test"
"beir@nfcorpus/test"
"beir@scidocs"
"beir@trec-covid"
"beir@webis-touche2020/v2"
)

dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
BENCHMARK=$(echo $dataset | cut -d'@' -f1)
SUBSET=$(echo $dataset | cut -d'@' -f2)
NAME=${SUBSET%%/*}
DATASET=${BENCHMARK}/${SUBSET}
MODEL_DIR=Llama-3.3-70B-Instruct
METRIC=nDCG@10

AUTOQREL_ROOT=${HOME}/APRIL/autoqrels
EVAL_RESULTS_DIR=${HOME}/APRIL/autoqrels-eval-results-same-retrieval/${NAME}

RETRIEVALS=(bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small)
LLM_RERANKERS=(judge judge_expr point umbrela rankgpt setmaxheaptopk  umbrela_rankgpt umbrela_setmaxheaptopk)
SUPERVISED_RERANKERS=(rankfirst rankzephyr)

# ── Judges: <retrieval>-rerank-<judge> runs, used as auto-qrel sources ───────
JUDGES=(judge judge_expr point rankgpt setmaxheaptopk umbrela umbrela_rankgpt umbrela_setmaxheaptopk)
STRATEGIES=(all)   # passed to output_autoqrel.py --strategies

judge_run_path() { echo "${HOME}/APRIL/runs/${MODEL_DIR}/run.${BENCHMARK}.$1-rerank-$2.${NAME}.txt"; }  # $1=retrieval $2=judge
autoqrel_dir()   { echo "${AUTOQREL_ROOT}/$1-rerank-$2/${NAME}/"; }

# ── Candidates for one retrieval ($1): fills CANDIDATES with "path|name" ─────
set_candidates() {
    local r1=$1 r2
    CANDIDATES=("${HOME}/runs-and-qrels/runs/${BENCHMARK}/run.${BENCHMARK}.${r1}.${NAME}.txt|${r1}")
    for r2 in "${LLM_RERANKERS[@]}"; do
        CANDIDATES+=("${HOME}/APRIL/runs/${MODEL_DIR}/run.${BENCHMARK}.${r1}-rerank-${r2}.${NAME}.txt|${r1}-rerank-${r2}")
    done
    for r2 in "${SUPERVISED_RERANKERS[@]}"; do
        CANDIDATES+=("${HOME}/APRIL/runs/supervised/run.${BENCHMARK}.${r1}-rerank-${r2}.${NAME}.txt|${r1}-rerank-${r2}")
    done
}
