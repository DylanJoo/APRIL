# Configuraitons

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

AUTOQREL_ROOT=${HOME}/APRIL/autoqrels-quantile
EVAL_RESULTS_DIR=${HOME}/APRIL/autoqrels-eval-results-quantile/${NAME}

# ── Judges: reranker runs over the pool, used as auto-qrel sources ───────────
POOL=pool-55-systems-top10
JUDGES=(judge judge_expr point rankgpt setmaxheaptopk umbrela umbrela_rankgpt umbrela_setmaxheaptopk)
STRATEGIES=(quantile_binary)   # passed to output_autoqrel.py --strategies
QUANTILE_CUTOFFS=(0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.975)   # one autollmqrel.quantile_binary@<q>.txt per cutoff: score >= per-query q-quantile -> relevant

judge_run_path() { echo "${HOME}/APRIL/runs/${MODEL_DIR}/run.${BENCHMARK}.${POOL}-rerank-$1.${NAME}.txt"; }
autoqrel_dir()   { echo "${AUTOQREL_ROOT}/${POOL}-rerank-$1/${NAME}/"; }

# ── Candidates: the 55 systems being evaluated (same set that built the pool) ─
RETRIEVALS=(bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small)
LLM_RERANKERS=(judge judge_expr point umbrela rankgpt setmaxheaptopk  umbrela_rankgpt umbrela_setmaxheaptopk)
SUPERVISED_RERANKERS=(rankfirst rankzephyr)

CANDIDATES=()   # entries are "path|name"
for r in "${RETRIEVALS[@]}"; do
    CANDIDATES+=("${HOME}/runs-and-qrels/runs/${BENCHMARK}/run.${BENCHMARK}.${r}.${NAME}.txt|${r}")
done
for r1 in "${RETRIEVALS[@]}"; do
    for r2 in "${LLM_RERANKERS[@]}"; do
        CANDIDATES+=("${HOME}/APRIL/runs/${MODEL_DIR}/run.${BENCHMARK}.${r1}-rerank-${r2}.${NAME}.txt|${r1}-rerank-${r2}")
    done
    for r2 in "${SUPERVISED_RERANKERS[@]}"; do
        CANDIDATES+=("${HOME}/APRIL/runs/supervised/run.${BENCHMARK}.${r1}-rerank-${r2}.${NAME}.txt|${r1}-rerank-${r2}")
    done
done
