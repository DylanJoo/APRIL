# Shared settings for the auto-qrel pipeline. Sourced (after `cd $HOME/APRIL`) by:
#   output_autoqrel.sh  judge runs  -> auto-qrel files     (pipeline step 2)
#   eval_autoqrel.sh    candidates  -> nDCG vs. each qrel  (pipeline step 3)
#
#   JUDGES      LLM reranker runs over POOL; each becomes auto-qrels
#               (one file per thresholding strategy).
#   CANDIDATES  the 55 systems that built the pool; each is scored against the
#               human qrels and every auto-qrel.
#
# Outputs (per dataset NAME):
#   qrel-analysis/autoqrels/{POOL}-rerank-{judge}/{NAME}/autollmqrel.{strategy}.txt
#   qrel-analysis/eval_results/{NAME}/human.txt
#   qrel-analysis/eval_results/{NAME}/{POOL}-rerank-{judge}.autollmqrel.{strategy}.txt

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

AUTOQREL_ROOT=${HOME}/APRIL/qrel-analysis/autoqrels
EVAL_RESULTS_DIR=${HOME}/APRIL/qrel-analysis/eval_results/${NAME}

# ── Judges: reranker runs over the pool, used as auto-qrel sources ───────────
POOL=pool-55-systems-top10
JUDGES=(judge judge_expr point rankgpt setmaxheaptopk umbrela umbrela_rankgpt umbrela_setmaxheaptopk)
STRATEGIES=(all)   # passed to output_autoqrel.py --strategies

judge_run_path() { echo "${HOME}/APRIL/runs/${MODEL_DIR}/run.${BENCHMARK}.${POOL}-rerank-$1.${NAME}.txt"; }
autoqrel_dir()   { echo "${AUTOQREL_ROOT}/${POOL}-rerank-$1/${NAME}/"; }

# ── Candidates: the 55 systems being evaluated (same set that built the pool) ─
RETRIEVALS=(bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small)
LLM_RERANKERS=(judge judge_expr point rankgpt setmaxheaptopk umbrela umbrela_rankgpt umbrela_setmaxheaptopk)
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
