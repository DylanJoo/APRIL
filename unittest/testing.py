from pprint import pprint

from reranking.config_manager import ConfigManager
config = ConfigManager('src/reranking/configs/rankgpt_config.yaml').get_config()
# pprint(config)

# -------Testing utils-------
import copy
from reranking.utils import Result, RerankMode
def get_eaxmple_result():
    query = "How to improve cardiovascular health?"
    passages = [
        "1. Engaging in at least 150 minutes of moderate aerobic exercise weekly significantly boosts heart health.",
        "2. Eating a balanced diet rich in fruits, vegetables, and whole grains can improve cardiovascular function.",
        "3. Regular blood pressure monitoring helps in early detection and prevention of heart disease.",
        "4. Managing cholesterol levels through lifestyle changes and medication protects arteries from plaque buildup.",
        "5. Reducing sodium intake can lower blood pressure and reduce heart disease risk.",
        "6. Getting at least seven hours of quality sleep per night supports overall vascular function.",
        "7. Reducing alcohol consumption has a positive effect on many aspects of physical wellness.",
        "8. Spending time outdoors can improve mental health and promote light physical activity.",
        "9. Learning a musical instrument can help improve cognitive functions in adults.",
        "10. Visiting art galleries fosters creativity and social engagement.",
    ]
    pairs = []
    for i, passage in enumerate(passages):
        pairs.append({'docid': f"docid_{i}", 'score': float(1/ (i+1)), 'content_dict': passage})

    results = []
    results.append(Result(qid='qid_0', query=query, hits=pairs))
    results.append(Result(qid='qid_1', query=query, hits=copy.deepcopy(pairs)))

    temp = results[1].hits[0]
    results[1].hits[0] = results[1].hits[-1]
    results[1].hits[-1] = temp
    return results

example_result = get_eaxmple_result()

# -------Testing llm prompt builder -------
# config.rerank_mode = RerankMode.PAIRWISE
config.rerank_mode = RerankMode.RANK_GPT
from reranking.prompt_builder import PromptBuilder
builder = PromptBuilder(
    config=config,
    include_system_message=True,
    system_message="You are a helpful assistant.",
    variable_passages=True,
    use_alpha=False
)
prompts = builder.create_prompt_batched(
    results=example_result,
    rank_start=0,
    rank_end=20
)
print(prompts[0])
# ----------------------------------

## --- Unit testing for the RankParser ---
# from reranking.result_parser import ResultParser
# rankparser = ResultParser(rerank_mode=RerankMode.RANK_GPT)
# outputs = rankparser.parse_response(
#     response_texts=\
#             ["[10] > [9] > [2] > [4] > [5] > [6] > [7] > [8] > [3] > [1]", \
#             "[10] > [8] > [3] > [4] > [5] > [6] > [7] > [9] > [2] > [1]"],
#     results=example_result,
#     rank_start=0,
#     rank_end=20
# )
# print([h['docid'] for h in outputs[0].hits])
# print([h['docid'] for h in outputs[1].hits])

## --- Unit testing for the input_assembler --- 
# from reranking.input_assembler import BubbleSort
# assembler = BubbleSort(
#     model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
#     rerank_mode=RerankMode.RANK_GPT,
#     include_system_message=True,
#     system_message="You are a helpful assistant.",
#     context_size=4096,
#     window_size=20,
#     step_size=10,
#     backend='litellm',  # 'vllm' or 'litellm'
# )
# reranked_results = assembler.run(
#     init_results=example_result,
#     rank_start=0,
#     rank_end=20
# )
# print(reranked_results)

## --- Unit testing for the reranking wrapper --- 
# from reranking.config_manager import ConfigManager
# config = ConfigManager('reranking/configs/rankgpt_config.yaml').get_config()
# pprint(config)
#
# from reranking.wrapper import ModularReranker
# rankllm = ModularReranker(
#     config, 
#     system_message= "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"
# )
