import copy
from pprint import pprint
from reranking.utils import Result, PromptMode

query = "How to improve cardiovascular health?"
passages = [
    "1. Visiting art galleries fosters creativity and social engagement.",
    "2. Engaging in at least 150 minutes of moderate aerobic exercise weekly significantly boosts heart health.",
    "3. Eating a balanced diet rich in fruits, vegetables, and whole grains can improve cardiovascular function.",
    "4. Regular blood pressure monitoring helps in early detection and prevention of heart disease.",
    "5. Managing cholesterol levels through lifestyle changes and medication protects arteries from plaque buildup.",
    "6. Reducing sodium intake can lower blood pressure and reduce heart disease risk.",
    "7. Getting at least seven hours of quality sleep per night supports overall vascular function.",
    "8. Reducing alcohol consumption has a positive effect on many aspects of physical wellness.",
    "9. Spending time outdoors can improve mental health and promote light physical activity.",
    "10. Learning a musical instrument can help improve cognitive functions in adults.",
]
pairs = []
for i, passage in enumerate(passages):
    pairs.append({'docid': f"docid_{i}", 'score': float(1/ (i+1)), 'content': passage})

results = []
results.append(Result(qid='qid_0', query=query, hits=pairs))
results.append(Result(qid='qid_1', query=query, hits=copy.deepcopy(pairs)))

temp = results[1].hits[0]
results[1].hits[0] = results[1].hits[-1]
results[1].hits[-1] = temp
# pprint(results)

## --- Unit testing for the PromptFormatter ---
# from reranking.prompt_builder import PromptFormatter
# formatter = PromptFormatter(
#     model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
#     prompt_mode=PromptMode.RANK_GPT,
#     include_system_message=True,
#     system_message="You are a helpful assistant that provides concise and informative answers to health-related questions.",
#     variable_passages=True,
#     use_alpha=True
# )
# prompts, lengths = formatter.create_prompt_batched(
#     results=results,
#     rank_start=0,
#     rank_end=20
# )
# print(lengths)
# print(prompts[0])
#
# ## --- Unit testing for the RankParser ---
# from reranking.result_parser import RankParser
# rankparser = RankParser(prompt_mode=PromptMode.RANK_GPT)
# outputs = rankparser.parse_response(
#     response_texts=\
#             ["[10] > [9] > [2] > [4] > [5] > [6] > [7] > [8] > [3] > [1]", \
#             "[10] > [8] > [3] > [4] > [5] > [6] > [7] > [9] > [2] > [1]"],
#     results=results,
#     rank_start=0,
#     rank_end=20
# )
# print([h['docid'] for h in outputs[0].hits])
# print([h['docid'] for h in outputs[1].hits])

## --- Unit testing for the reranking wrapper --- 
from reranking.rankllm import RankListwiseLLM 
rankllm = RankListwiseLLM(
    model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
    prompt_mode=PromptMode.RANK_GPT,
    include_system_message=True,
    system_message="You are a helpful assistant.",
    context_size=4096,
    window_size=20,
    step_size=10,
)
reranked_results = rankllm.sliding_windows_batched(
    retrieved_results=results,
    rank_start=0,
    rank_end=20
)
pprint(reranked_results)
