import argparse
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams
import uuid

# cleaning
import contextlib
import gc
import torch
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel
)

class LLM:

    def __init__(
        self,
        model,
        temperature=0.0,
        top_p=1.0,
        logprobs=None,
        max_tokens=128,
        dtype='half',
        gpu_memory_utilization=0.9,
        num_gpus=1, 
        enforce_eager=False,
        max_model_len=20480,
    ):
        args = AsyncEngineArgs(
            model=model,
            dtype=dtype,
            enforce_eager=enforce_eager,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs.from_cli_args(args))

        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            skip_special_tokens=False,
            logprobs=logprobs
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def _iterate_over_output(self, output_iterator: AsyncStream) -> str:
        last_text = ""
        async for output in output_iterator:
            last_text = output.outputs[0].text
            # last_logprobs = output.outputs[0].logprobs
        return last_text

    def generate(self, prompts, prob=False, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        sampling_params = self.sampling_params
        sampling_params.min_tokens = kwargs.get('min_tokens', 1)

        if prob:
            return self.loop.run_until_complete(self._agenerate_prob(prompts, sampling_params))
        else:
            return self.loop.run_until_complete(self._agenerate_text(prompts, sampling_params))

    async def _agenerate_text(self, prompts, sampling_params):
        request_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]
        
        # Add requests to the engine
        output_iterators = [
            await self.engine.add_request(request_id, prompt, sampling_params)
            for request_id, prompt in zip(request_ids, prompts)
        ]
        
        # Gather all the outputs
        outputs = await asyncio.gather(*[
            self._iterate_over_output(output_iterator)
            for output_iterator in output_iterators
        ])
        return list(outputs)

    async def _agenerate_prob(self, prompts: List[str]) -> List[float]:

        # singlge function call of selected token prob
        def _generate_prob(prompt: str) -> float:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                logprobs=self.logprobs,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            # dict of scores: {first token: first token logprob}
            tok_logps = response.choices[0].logprobs.top_logprobs[0] 
            yes_ = math.exp(max(
                [-1e2] + [
                    logp for tok, logp in tok_logps.items() 
                    if tok in self.yes_tokens
                ]
            ))
            no_ = math.exp(max(
                [-1e2] + [
                    logp for tok, logp in tok_logps.items() 
                    if tok in self.no_tokens 
                ]
            ))
            score = yes_ / (no_ + yes_)
            return score

        # Gather all the outputs
        outputs = await asyncio.gather(*[
            asyncio.to_thread(_generate_prob, prompt) for prompt in prompts
        ])
        return list(outputs)
