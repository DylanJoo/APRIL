# -------Testin llm provider -------
from reranking.llm_provider.vllm_api import LLM
llm = LLM(model_name_or_path='Qwen/Qwen3-1.7B', temperature=0.0, top_p=1.0, logprobs=20, max_tokens=1)
llm.set_classification(
    yes_strings=[' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES'],
    no_strings=[' No', 'No', ' no', 'no', 'NO', ' NO'],
    id_strings=[chr(i) for i in range(65, 85)]  # A to J
)

# from reranking.llm_provider.litellm_api import LLM
# llm = LLM(model='llama3.3-70b-instruct', temperature=0.0, top_p=1.0, logprobs=20, max_tokens=1)
# llm.set_classification()
# result = llm.generate([f'write a poem of {i}.' for i in range(10)], prob=True)

# Binary prob 
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
result = llm.generate(prompts, binary_probs=True)
print(result)
# >>> Yes # Yes # Yes # No # Yes # No # Yes # Yes # Yes # Yes
# ----------------------------------

# Distribution prob
# Change the testing to document relevance
prompts = """\
Query: Is Paris the capital of France?
Documents:
[A] Paris is the capital city of France.
[B] The Eiffel Tower is located in Paris.
[C] France is known for its rich history and culture.
[D] The Louvre Museum is in Paris.
[E] Paris is famous for its cuisine and fashion.
[F] The Seine River runs through Paris.
[G] Paris is a major European city.
[H] The capital of France is Paris.
[I] Paris has many historical landmarks.
[J] The French Revolution began in Paris.

Which statement is the most relevant to the query? Write the alphabet enclose with bracket. """

result = llm.generate(prompts, dist_logp=True)
print(result)
# >>> [logp_'1', logp_'2', logp_'3', logp_'4', logp_'5', ...]
# ----------------------------------

