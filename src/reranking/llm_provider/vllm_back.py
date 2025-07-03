"""
Update vllm engine with chat template.
The chat template is to be implemented.
"""
import torch
import math
import vllm
from typing import List

class LLM:

    def __init__(
        self,
        model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
        temperature=0.0,
        top_p=1.0,
        logprobs=20,
        max_tokens=10,
        num_gpus=1, 
        dtype='half', 
        max_model_len=32768,
        gpu_memory_utilization=0.9, 
        **kwargs
    ):
        self.model = vllm.LLM(
            model_name_or_path,
            max_logprobs=30,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=num_gpus,
        )
        self.sampling_params = vllm.SamplingParams(
            temperature=temperature, 
            skip_special_tokens=False,
            logprobs=logprobs,
            prompt_logprobs=None,
            max_tokens=max_tokens, 
            min_tokens=2, 
        )
        self.tokenizer = self.model.get_tokenizer()
        self.yes_tokens = None
        self.no_tokens = None

    def set_classification(self, yes_strings, no_strings):
        """ vLLM outputs probabilties of each token ids """
        self.yes_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in yes_strings]
        self.no_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in no_strings]

    def generate(self, prompts, prob=False, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]

        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)

        if prob is False:
            return [o.outputs[0].text for o in outputs]

        tok_logps = [o.outputs[0].logprobs for o in outputs]

        scores = []
        for tok_logp in tok_logps:
            yes_ = math.exp(max( 
                [-1e2] + [
                    tok_logp[0][i].logprob for tok_id in tok_logp[0] 
                    if tok_id in self.yes_tokens
                ] 
            ))
            no_ = math.exp(max( 
                [-1e2] + [
                    tok_logp[0][i].logprob for tok_id in tok_logp[0] 
                    if tok_id in self.no_tokens
                ]
            ))
            scores.append( (yes_) / (no_ + yes_) )

        return scores
