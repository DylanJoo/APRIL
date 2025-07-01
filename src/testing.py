from reranking.llm_provider.utils import cleanup_vllm

# from reranking.llm_provider.vllm_api import LLM
# llm = LLM(model='Qwen/Qwen3-1.7B', temperature=0.0, top_p=1.0, logprobs=20, max_tokens=1)
# cleanup_vllm(llm)

from reranking.llm_provider.litellm_api import LLM
llm = LLM(model='llama3.3-70b-instruct', temperature=0.0, top_p=1.0, logprobs=20, max_tokens=1)
llm.set_classification()
# result = llm.generate([f'write a poem of {i}.' for i in range(10)], prob=True)

prompts = [
"Is Paris the capital of France?",
"Does water boil at 100°C?",
"Is the sun a star?",
"Can humans breathe underwater unaided?",
"Is Mount Everest the tallest mountain?",
"Do whales lay eggs?",
"Is Mars known as the Red Planet?",
"Was Albert Einstein a physicist?",
"Is gold a metal?",
"Does the Earth orbit the sun?",
]
# Yes # Yes # Yes # No # Yes # No # Yes # Yes # Yes # Yes
result = llm.generate(prompts, prob=True)

print(result)

