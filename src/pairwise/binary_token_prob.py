import csv
import os
import logging
import json
import math
from pairwise.rank_llm.rankllm import PromptMode, RankPairwiseOSLLM
# from pairwise.rank_llm.reranker import Reranker
# from pairwise.rank_llm.rank_listwise_os_llm import RankListwiseOSLLM
from pairwise.rank_llm.utils import Result
from utils.tools import convert_run_to_result

def rerank(
    model, 
    run, queries, corpus,
    use_logits, 
    use_alpha, 
    variable_passages,
    top_k, 
    window_size, 
    step_size, 
    batched, 
    context_size, 
    rerank_type="text", 
):
    system_message = \
    "You are CompareLLM, an intelligent assistant that can compare passages based on their relevancy to the query."

    results_for_rerank = convert_run_to_result(run, queries, corpus)

    # Initialize the ranking model
    agent = RankPairwiseOSLLM(
        model=model,
        context_size=context_size,
        prompt_mode=PromptMode.RANK_GPT,
        num_few_shot_examples=0,
        device="cuda",
        num_gpus=1,
        variable_passages=variable_passages,
        window_size=window_size,
        system_message=system_message,
        batched=batched,
        rerank_type=rerank_type,
    )

    ## ps
    if args.batch_size > 1:
        results = agent.sliding_windows_batched(
            results_for_rerank,
            use_logits=False, # set to True if comparing label's probabilities
            use_alpha=use_alpha,
            rank_start=0,
            rank_end=min(rank_end, len(retrieved_results[0].hits)), #TODO: Fails arbitrary hit sizes
            window_size=window_size,
            step=step,
            logging=logging,
        )

    reranked_run = {}
    for result in reranked_results:
        reranked_run[result.qid] = {}
        for rank, hit in enumerate(result.hits, start=1):
            hit['rank'] = rank
            reranked_run[result.qid].update({ hit['docid']: hit['score'] })

    return reranked_run
