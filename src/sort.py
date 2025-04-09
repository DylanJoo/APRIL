"""
Script to rerank with APRIL-partition-sort
"""
from llm.vllm_back import LLM
import loader
from prompts import PromptMode

logger = logging.getLogger(__name__)

def main(
    model_name_or_path: str,
    ir_datasets_name: str,
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
    **kwargs,
):
    corpus, queries, qrels = loader.load(ir_datasets_name, query_fields, doc_fields)

    model = LLM(
        model_name=model_name_or_path,
        max_length=kwargs.pop('max_length', 512),
        batch_size=kwargs.pop('batch_size', 16),
        device=kwargs.pop('device', 'cuda'),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--ir_datasets_name', type=str)
    parser.add_argument('--query_fields', type=str, nargs='+')
    parser.add_argument('--doc_fields', type=str, nargs='+')
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()

    main()


# kwargs: 
# ("prompt_mode", PromptMode.MONOT5),
# ("context_size", 512),
# ("device", "cuda"),
# ("batch_size", 64),
# ("context_size", 4096),
# ("prompt_mode", PromptMode.RANK_GPT),
# ("num_few_shot_examples", 0),
# ("device", "cuda"),
# ("num_gpus", 1),
# ("variable_passages", False),
# ("window_size", 20),
# ("system_message", None),
# ("sglang_batched", False),
# ("tensorrt_batched", False),
# ("use_logits", False),
# ("use_alpha", False),
