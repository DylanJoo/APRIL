from reranking.prompt_builder import PromptBuilder
from reranking.config_manager import ConfigManager

config = ConfigManager(
    rerank_mode='April',
    top_k=100,
    rank_start=0,
    rank_end=100,
    llm={'max_model_len': 8192, 'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct'}
).get_config()

p = PromptBuilder(config=config, include_system_message='{system_message}')
rank_start = config.rank_start - config.window_size
rank_end = config.rank_end
prefix = p.formatter.prefix("XXX", doc_list=[f'document {i}' for i in range(100)][rank_start:rank_end])
body = p.formatter.body("XXX", doc_list=[f'document {i}' for i in range(100)][rank_start:rank_end])
postfix = p.formatter.postfix("XXX", doc_list=[f'document {i}' for i in range(100)][rank_start:rank_end])

print(prefix)
print(body[0] if isinstance(body, list) else body)
print(postfix[0] if isinstance(postfix, list) else postfix)

print('=======the next one======')
if isinstance(prefix, list):
    print(prefix[1])
if isinstance(postfix, list):
    print(postfix[1])
