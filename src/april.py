""" 
Script to rerank with APRIL 
"""
from llm.vllm_back import LLM
import loader
from prompts import PromptMode

logger = logging.getLogger(__name__)

def main(
    model_name_or_path: str,
    **kwargs,
):
    model = LLM(
        model_name=model_name_or_path,
        max_length=kwargs.pop('max_length', 512),
        batch_size=kwargs.pop('batch_size', 16),
        device=kwargs.pop('device', 'cuda'),
    )

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
