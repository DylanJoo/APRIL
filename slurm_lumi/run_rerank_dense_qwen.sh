#!/bin/bash -l
<<<<<<< HEAD
#SBATCH --job-name=dense-qwen
=======
<<<<<<<< HEAD:slurm_lumi/run_rerank_dense_qwen3.sh
#SBATCH --job-name=dense-qwen2
========
#SBATCH --job-name=dense-qwen
>>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f:slurm_lumi/run_rerank_dense_qwen.sh
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
#SBATCH --partition=small-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --mem=256G
#SBATCH --nodes=1
<<<<<<< HEAD
=======
<<<<<<<< HEAD:slurm_lumi/run_rerank_dense_qwen3.sh
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
#SBATCH --array=0
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --account=project_465002532
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

<<<<<<< HEAD
=======
module --force purge
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
========
#SBATCH --array=0-6
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --time=72:00:00             # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002532 # Project for billing
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
export HIP_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1 
export VLLM_SKIP_P2P_CHECK=1
<<<<<<< HEAD
=======
>>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f:slurm_lumi/run_rerank_dense_qwen.sh
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f

cd $HOME/APRIL
MODEL=Qwen/Qwen2.5-72B-Instruct
mkdir -p runs/${MODEL##*/}

<<<<<<< HEAD
=======
MODEL=Qwen/Qwen2.5-72B-Instruct
export HIP_VISIBLE_DEVICES=0,1,2,3,4

>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
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
benchmark=$(echo $dataset | cut -d'@' -f1)
subset=$(echo $dataset | cut -d'@' -f2)

## POINTWISE
<<<<<<< HEAD
needs_pointwise=false
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
    for method in judge judge_expr point; do
        output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
        if [ ! -f "$output_run" ]; then
            needs_pointwise=true
            break 2
        fi
    done
done

if [ "$needs_pointwise" = true ]; then
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --disable-custom-all-reduce \
    --max-model-len 10240 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 > vllm_server_qwen.log 2>&1 &
=======
<<<<<<<< HEAD:slurm_lumi/run_rerank_dense_qwen3.sh
singularity exec $SIF_QWEN \
    python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --disable-custom-all-reduce \
    --disable-log-stats \
    --enforce-eager \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 > vllm_server_qwen.log 2>&1 &
========
# srun singularity exec $SIF_QWEN \
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --disable-custom-all-reduce \
    --disable-log-stats \
    --max-model-len 10240 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 > vllm_server_qwen.log 2>&1 &
>>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f:slurm_lumi/run_rerank_dense_qwen.sh
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for method in judge judge_expr point; do
<<<<<<< HEAD
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
=======
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt 
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
<<<<<<< HEAD
    srun singularity exec $SIF \
=======
    singularity exec $SIF_QWEN \
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --llm.backend=request \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL
done
done
kill $PID
<<<<<<< HEAD
fi

## SETWISE
method=setmaxheaptopk
needs_setwise=false
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ ! -f "$output_run" ]; then
        needs_setwise=true
        break
    fi
done

if [ "$needs_setwise" = true ]; then
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --disable-custom-all-reduce \
    --max-model-len 20480 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 > vllm_server_qwen.log 2>&1 &
=======

## SETWISE
singularity exec $SIF_QWEN \
    python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --disable-custom-all-reduce \
    --disable-log-stats \
    --enforce-eager \
    --max-model-len 20480 \
    --dtype bfloat16 \
<<<<<<<< HEAD:slurm_lumi/run_rerank_dense_qwen3.sh
    --tensor-parallel-size 8 > vllm_server_qwen.log 2>&1 &
========
    --disable-custom-all-reduce \
    --tensor-parallel-size 4 > vllm_server_qwen.log 2>&1 &
>>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f:slurm_lumi/run_rerank_dense_qwen.sh
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

<<<<<<< HEAD
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
=======
method=setmaxheaptopk
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt 
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
<<<<<<< HEAD
    srun singularity exec $SIF \
=======
    singularity exec $SIF_QWEN \
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --llm.backend=request \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL
done
kill $PID
<<<<<<< HEAD
fi

## LISTWISE
method=rankgpt
needs_listwise=false
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ ! -f "$output_run" ]; then
        needs_listwise=true
        break
    fi
done

if [ "$needs_listwise" = true ]; then
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --disable-custom-all-reduce \
    --max-model-len 30720 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 > vllm_server_qwen.log 2>&1 &
=======

## LISTWISE
singularity exec $SIF_QWEN \
    python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
<<<<<<<< HEAD:slurm_lumi/run_rerank_dense_qwen3.sh
    --disable-custom-all-reduce \
    --disable-log-stats \
    --enforce-eager \
    --max-model-len 30720 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 > vllm_server_qwen.log 2>&1 &
========
    --max-model-len 30720 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 4 > vllm_server_qwen.log 2>&1 &
>>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f:slurm_lumi/run_rerank_dense_qwen.sh
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

<<<<<<< HEAD
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
=======
method=rankgpt
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt 
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
<<<<<<< HEAD
    srun singularity exec $SIF \
=======
    singularity exec $SIF_QWEN \
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --llm.backend=request \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL
done
kill $PID
<<<<<<< HEAD
fi
=======
>>>>>>> 5cb398d90983f53b5264df8f4e6fc8f48cff9e7f
