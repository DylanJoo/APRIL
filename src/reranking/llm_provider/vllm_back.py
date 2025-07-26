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
        max_model_len=10240,
        gpu_memory_utilization=0.9, 
        **kwargs
    ):
        self.model = vllm.LLM(
            model_name_or_path,
            dtype=dtype,
            enforce_eager=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True
        )
        self.sampling_params = vllm.SamplingParams(
            temperature=temperature, 
            skip_special_tokens=False,
            logprobs=logprobs,
            prompt_logprobs=None,
            max_tokens=max_tokens, 
            min_tokens=1, 
        )
        self.tokenizer = self.model.get_tokenizer()
        self.yes_tokens = None
        self.no_tokens = None

    def set_classification(
        self, 
        yes_strings=[' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES'],
        no_strings=[' No', 'No', ' no', 'no', 'NO', ' NO']
    ):
        """ vLLM outputs probabilties of each token ids """
        self.yes_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in yes_strings]
        self.no_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in no_strings]

    # self.engine.reset_prefix_cache
    def generate(self, prompts, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]

        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)
        tok_logps = [o.outputs[0].logprobs for o in outputs]

        scores = []
        for tok_logp in tok_logps:
            yes_ = math.exp(max( 
                [-1e2] + [
                    v.logprob for k, v in tok_logp[0].items()
                    if k in self.yes_tokens
                ] 
            ))
            no_ = math.exp(max( 
                [-1e2] + [
                    v.logprob for k, v in tok_logp[0].items()
                    if k in self.no_tokens
                ]
            ))
            scores.append( (yes_) / (no_ + yes_) )

        return scores
