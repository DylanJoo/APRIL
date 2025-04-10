import loader
import logging
from typing import Optional

from pointwise import pt_rerank
from llm.vllm_back import LLM
from utils.tools import load_runs

logger = logging.getLogger(__name__)

def main(
    model_name_or_path: str,
    run_path: str, 
    ir_datasets_name: str,
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
    **kwargs,
):
    run = load_runs(run_path, topk=10)
    corpus, queries, qrels = loader.load(
        ir_datasets_name, query_fields, doc_fields
    )

    model = LLM(
        model=model_name_or_path,
        temperature=0, 
        top_p=1.0,
        prompt_logprobs=5,
        gpu_memory_utilization=0.9, 
    )

    reranked_run = qlm_rerank(
        model=model,
        run=run,
        queries=queries,
        corpus=corpus,
        batch_size=8
    )

    # evaluation
    result = ir_measures.calc_aggregate([nDCG@10], qrels, reranked_run)[nDCG@10]
    print(result)

main(
    model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    run_path="/home/jju/APRIL/runs/run.msmarco-v1-passage.bm25-rm3-default.dl19.txt",
    ir_datasets_name='msmarco-passage/trec-dl-2019',
)

# kwargs: 
# ("prompt_mode", PromptMode.MONOT5),
# ("context_size", 512),
# ("device", "cuda"),
# ("batch_size", 64),
# ("context_size", 4096),
# ("prompt_mode", PromptMode.RANK_GPT),
# ("num_few_shot_examples", 0),
# ("device", "cuda"),
# ("num_gpus", 1),
# ("variable_passages", False),
# ("window_size", 20),
# ("system_message", None),
# ("sglang_batched", False),
# ("tensorrt_batched", False),
# ("use_logits", False),
# ("use_alpha", False),
