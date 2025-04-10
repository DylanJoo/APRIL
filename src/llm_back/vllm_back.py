import vllm
from typing import List

class LLM:

    def __init__(self, 
        model, 
        temperature=0.7, top_p=0.9, 
        dtype='half', gpu_memory_utilization=0.75, 
        num_gpus=1, 
        think_activated=False,
        max_model_len=13712
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
            skip_special_tokens=False
        )
        self.think_activated = think_activated

    def postprocess(self, outputs: List):
        if self.think_activated:
            for i, o in enumerate(outputs):
                if '</think>' in o:
                    outputs[i] = o.split('</think>')[-1]
        return outputs

    def generate(self, prompts, max_tokens=256, min_tokens=32):
        self.sampling_params.max_tokens = max_tokens
        self.sampling_params.min_tokens = min_tokens

        if isinstance(prompts, str):
            prompts = [prompts]

        outputs = self.model.generate(x, self.sampling_params, use_tqdm=False)
        if len(outputs) > 1:
            return self.postprocess([o.outputs[0].text for o in outputs])
        else:
            return self.postprocess([outputs[0].outputs[0].text])

