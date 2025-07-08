from reranking.prompt_builder import PromptBuilder
from reranking.config_manager import ConfigManager
from reranking.utils import RerankMode

config = ConfigManager(
    rerank_mode='Pairwise',
    top_k=100,
    rank_end=10,
    llm={'max_model_len': 8192, 'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct'}
).get_config()
rerank_mode = RerankMode(config.rerank_mode)

p = PromptBuilder(config=config, rerank_mode=rerank_mode, include_system_message='{system_message}')
print(p.formatter.prefix("XXX"))
print(p.formatter.body("XXX", doc_list=[f'{i} document' for i in range(10)])[0])
print(p.formatter.postfix("XXX"))
