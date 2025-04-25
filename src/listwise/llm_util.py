"""
This code is modified from RankFirst repo using rank-llm
https://github.com/gangiswag/llm-reranker/blob/main/scripts/utils/llm_util.py
"""
import csv
import os
import logging
import json
import math
from utils.result import Result, ResultsLoader
from utils.rankllm import PromptMode, RankLLM
from utils.reranker import Reranker
from utils.rank_listwise_os_llm import RankListwiseOSLLM

def rerank_beir_outputs_llm(model, results_for_rerank, use_logits, use_alpha, top_k, window_size, step_size, batched, context_size, rerank_type="text", code_prompt_type="docstring"):
    """
    Rerank outputs using either text or code reranking
    
    Args:
        rerank_type (str): Whether to perform "text" or "code" reranking
        code_prompt_type (str): For code reranking, whether to use "docstring" or "github_issue" prompts
    """
    # Validate parameters for code reranking
    if rerank_type == "code":
        if use_logits or use_alpha:
            print("Warning: Code reranking does not support logits or alpha mode. These will be disabled.")
            use_logits = False
            use_alpha = False

    # Select appropriate system message based on rerank type and prompt type
    if rerank_type == "code":
        if code_prompt_type == "docstring":
            system_message = "You are CodeRanker, an intelligent code reviewer that can analyze doc strings and rank code snippets based on their relevance to the doc string."
        elif code_prompt_type == "github_issue": 
            system_message = "You are CodeRanker, an intelligent code reviewer that can analyze GitHub issues and rank code functions based on their relevance to contain the faults causing the GitHub issue."
        else:
            raise ValueError(f"Invalid code_prompt_type: {code_prompt_type}")
    else:  # text reranking
        system_message = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"

    # Initialize the ranking model
    agent = RankListwiseOSLLM(
        model=model,
        context_size=context_size,
        prompt_mode=PromptMode.RANK_GPT,
        num_few_shot_examples=0,
        device="cuda",
        num_gpus=1,
        variable_passages=True,
        window_size=window_size,
        system_message=system_message,
        batched=batched,
        rerank_type=rerank_type,
        code_prompt_type=code_prompt_type
    )

    # Perform reranking
    reranker = Reranker(agent=agent)
    reranked_results = reranker.rerank(
        retrieved_results=results_for_rerank,
        use_logits=use_logits,
        use_alpha=use_alpha,
        rank_start=0,
        rank_end=top_k,
        window_size=window_size,
        step=step_size,
        logging=False,
        batched=batched
    )

    for result in reranked_results:
        for rank, hit in enumerate(result.hits, start=1):
            hit['rank'] = rank

    return reranked_results
