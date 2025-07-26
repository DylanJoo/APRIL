from reranking.prompt_builder import PromptBuilder
from reranking.config_manager import ConfigManager

config = ConfigManager(
    rerank_mode='SetTopK',
    top_k=100,
    rank_start=0,
    rank_end=100,
    llm={'max_model_len': 8192, 'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct'}
).get_config()

p = PromptBuilder(config=config, include_system_message='{system_message}')
rank_start = config.rank_start - config.window_size
rank_end = config.rank_end

inputs = {
    'query': 'XXX',
    'doc_list': [f'document {i}' for i in range(5)][rank_start:rank_end],
    'idx_pairs': [(0, 1, 2), (3, 4, 5)]
}
    # 'idx_pairs': [(0, 1), (2, 3)]


prefix = p.formatter.prefix(**inputs)
body = p.formatter.body(**inputs)
postfix = p.formatter.postfix(**inputs)

print(f"Prefix: {prefix}")
if isinstance(body, list):
    body = "\n".join(body)
    print(f"Body (list): {body}")
else:
    print(f"Body: {body}")

if isinstance(postfix, list):
    postfix = "\n".join(postfix)
    print(f"Postfix (list): {postfix}")
else:
    print(f"Postfix: {postfix}")

# print(prefix)
# print(body[0] if isinstance(body, list) else body)
# print(postfix[0] if isinstance(postfix, list) else postfix)
#
# print('=======the next one======')
# if isinstance(prefix, list):
#     print(prefix[1])
# if isinstance(body, list):
#     print(body[1])
# if isinstance(postfix, list):
#     print(postfix[1])
