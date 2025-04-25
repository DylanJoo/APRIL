# from vllm import LLM, SamplingParams
# from utils.tools import batch_iterator
#
# # model, sampling_params
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, prompt_logprobs=True, max_tokens=1)
# model = LLM(model="facebook/opt-125m")
# tokenizer = model.get_tokenizer()
#
# # prompt preparation
# template = "Please write a question based on this passage.\nPassage: This is testing document.\nQuestion: This is a testing query."
#
# all_tokens = tokenizer(template)
# prompt_tokens = tokenizer(' This is a testing query.', add_special_tokens=False)['input_ids']
#
# id_pairs, prompts = [], []
# prompts.append( template )
#
# # inference
# batch = prompts * 5
# outputs = model.generate(batch, sampling_params)
# for output in outputs:
#     logprobs = [output.prompt_logprobs[i][id].logprob if i > 0 else -1 for i, id in enumerate(output.prompt_token_ids)][-len(prompt_tokens):]
#     print(tokenizer.tokenize(' This is a testing query.'))
#     print(prompt_tokens)
#     print(logprobs)

# doc = "An AI model is a computer program that uses algorithms and data to perform tasks that typically require human intelligence, such as understanding natural language, recognizing patterns, and making decisions."
# query = "What is an AI model?"
# template = f"Passage: {doc}\nQuery: {query}\nIs this passage relevant to the query? Please answer Yes or No.\nAnswer:"
# from llm.hf_encode import LLM
# model = LLM(model="meta-llama/Llama-3.2-1B-Instruct", model_class='clm', temperature=0)
# model.inference(x=[template, template], max_tokens=1)

# module load 2024
# module load CUDA/12.6.0

import loader
from utils.tools import load_runs
import ir_measures
from ir_measures import *
run_path=f"/home/jju/APRIL/runs/run.msmarco-v1-passage.bm25-default.dl19.txt"
ir_datasets_name='msmarco-passage/trec-dl-2019/judged'

run = load_runs(run_path, topk=100, output_score=True)
_, _, qrels = loader.load(ir_datasets_name)
optimal_run = {}
for qid in run:
    optimal_run[qid] = {}
    for i, docid in enumerate(qrels[qid]):
        if docid in run[qid]:
            score = qrels[qid][docid] + (1/(i+1))
        else:
            score = (1/(i+1))
        optimal_run[qid].update({docid: score})

print(ir_measures.calc_aggregate([nDCG@10, nDCG@20], qrels, optimal_run))
