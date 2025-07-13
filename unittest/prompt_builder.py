from reranking.prompt_builder import PromptBuilder
from reranking.config_manager import ConfigManager

config = ConfigManager(
    rerank_mode='April',
    top_k=100,
    rank_end=10,
    llm={'max_model_len': 8192, 'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct'}
).get_config()

p = PromptBuilder(config=config, include_system_message='{system_message}')

prefix = p.formatter.prefix("XXX", doc_list=[f'document {i}' for i in range(100)])
body = p.formatter.body("XXX", doc_list=[f'document {i}' for i in range(100)], rank_end=4)
postfix = p.formatter.postfix("XXX", doc_list=[f'document {i}' for i in range(3)])


print(prefix)
print(body[0] if isinstance(body, list) else body)
print(postfix[0] if isinstance(postfix, list) else postfix)
