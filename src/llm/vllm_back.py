import math
import vllm
from typing import List

class LLM:

    def __init__(self, 
        model, 
        temperature=0.7, top_p=0.9, 
        dtype='half', gpu_memory_utilization=0.75, 
        num_gpus=1, 
        max_model_len=13712,
        logprobs=None,
        prompt_logprobs=None
    ):
        self.model = vllm.LLM(
            model, 
            dtype=dtype,
            enforce_eager=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )
        self.sampling_params = vllm.SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            skip_special_tokens=False,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs
        )

        self.logprobs = logprobs
        self.prompt_logprobs = prompt_logprobs

    def postprocess(self, outputs: List):
        return outputs

    def generate(
        self, 
        prompts, max_tokens=256, min_tokens=32, 
        query_tokens=None,
        yes_tokens=None,
        no_tokens=None
    ):
        self.sampling_params.max_tokens = max_tokens
        self.sampling_params.min_tokens = min_tokens

        if isinstance(prompts, str):
            prompts = [prompts]

        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)
        texts = self.postprocess([o.outputs[0].text for o in outputs])

        if (yes_tokens is not None):
            prompt_logprobs = [o.outputs[0].logprobs for o in outputs]
            scores = []
            for p in prompt_logprobs:
                yes_ = math.exp(max( [-1e9]+[p[0][i].logprob for i in yes_tokens if (i in p[0])] ))
                no_ = math.exp(max( [-1e9]+[p[0][i].logprob for i in no_tokens if (i in p[0])] ))
                scores.append( (yes_) / (no_ + yes_) )
        else:
            scores = None

        return texts, scores

