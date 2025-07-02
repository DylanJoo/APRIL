import math
import argparse
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
import uuid
from typing import List

class LLM:

    def __init__(
        self,
        model_name_or_path: str,
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
            model=model_name_or_path,
            dtype=dtype,
            enforce_eager=enforce_eager,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )
        self.model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs.from_cli_args(args))

        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            skip_special_tokens=False,
            logprobs=logprobs
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.yes_tokens = None
        self.no_tokens = None
        if logprobs:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def set_classification(
        self, 
        yes_strings=[' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES'],
        no_strings=[' No', 'No', ' no', 'no', 'NO', ' NO']
    ):
        self.yes_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in yes_strings]
        self.no_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in no_strings]

    async def _iterate_over_output(self, output_iterator: AsyncStream, logprobs=False) -> str:
        last_text = ""
        async for output in output_iterator:
            last_text = output.outputs[0].text

            if logprobs:
                tok_logps = output.outputs[0].logprobs[0]
                yes_ = math.exp(max(
                    [-1e2] + [
                        item.logprob for tok, item in tok_logps.items() 
                        if tok in self.yes_tokens
                    ]
                ))
                no_ = math.exp(max(
                    [-1e2] + [
                        item.logprob for tok, item in tok_logps.items() 
                        if tok in self.no_tokens 
                    ]
                ))
                score = yes_ / (no_ + yes_)
                return score

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
            await self.model.add_request(request_id, prompt, sampling_params)
            for request_id, prompt in zip(request_ids, prompts)
        ]
        
        # Gather all the outputs
        outputs = await asyncio.gather(*[
            self._iterate_over_output(output_iterator)
            for output_iterator in output_iterators
        ])
        return list(outputs)

    async def _agenerate_prob(self, prompts, sampling_params) -> List[float]:
        request_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]
        
        # Add requests to the engine
        output_iterators = [
            await self.model.add_request(request_id, prompt, sampling_params)
            for request_id, prompt in zip(request_ids, prompts)
        ]
        
        # Gather all the outputs
        outputs = await asyncio.gather(*[
            self._iterate_over_output(output_iterator, logprobs=True)
            for output_iterator in output_iterators
        ])
        return list(outputs)

    # async def _agenerate_prob(self, prompts, sampling_params) -> List[float]:
    #
    #     # singlge function call of selected token prob
    #     def _generate_prob(prompt: str) -> float:
    #         response = self.client.completions.create(
    #             model=self.model,
    #             prompt=prompt,
    #             logprobs=self.logprobs,
    #             temperature=self.temperature,
    #             top_p=self.top_p,
    #             max_tokens=self.max_tokens,
    #         )
    #
    #         # dict of scores: {first token: first token logprob}
    #         tok_logps = response.choices[0].logprobs.top_logprobs[0] 
    #         yes_ = math.exp(max(
    #             [-1e2] + [
    #                 logp for tok, logp in tok_logps.items() 
    #                 if tok in self.yes_tokens
    #             ]
    #         ))
    #         no_ = math.exp(max(
    #             [-1e2] + [
    #                 logp for tok, logp in tok_logps.items() 
    #                 if tok in self.no_tokens 
    #             ]
    #         ))
    #         score = yes_ / (no_ + yes_)
    #         return score
    #
    #     # Gather all the outputs
    #     outputs = await asyncio.gather(*[
    #         asyncio.to_thread(_generate_prob, prompt) for prompt in prompts
    #     ])
    #     return list(outputs)

    # def inference(self, prompts):
    #     if isinstance(prompts, str):
    #         prompts = [prompts]
    #
    #     outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)
    #     tok_logps = [o.outputs[0].logprobs for o in outputs]
    #
    #     scores = []
    #     for tok_logp in tok_logps:
    #         yes_ = math.exp(max( 
    #             [-1e2] + [
    #                 tok_logp[0][i].logprob for tok_id in tok_logp[0] 
    #                 if tok_id in self.yes_tokens
    #             ] 
    #         ))
    #         no_ = math.exp(max( 
    #             [-1e2] + [
    #                 tok_logp[0][i].logprob for tok_id in tok_logp[0] 
    #                 if tok_id in self.no_tokens
    #             ]
    #         ))
    #         scores.append( (yes_) / (no_ + yes_) )
    #
    #     return scores
