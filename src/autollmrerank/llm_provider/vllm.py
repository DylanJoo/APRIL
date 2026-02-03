# TODO: setup different style of token `id` categories
# TODO: in addition to that, also allow use to save token's `term` or `id`.
import math
import argparse
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
import uuid
import re
from typing import List, Optional
import logging
logger = logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)

class LLM:

    def __init__(
        self,
        model_name_or_path: str = 'meta-llama/Llama-3.2-1B-Instruct',
        temperature=0.0,
        top_p=1.0,
        logprobs=None,
        max_tokens=128,
        dtype='half',
        gpu_memory_utilization=0.9,
        num_gpus=1, 
        max_model_len=10240,
        **kwargs
    ):
        print(f"Unused kwargs: {kwargs}")
        """
        # AMPERE GPU: dtype='float16', enable_prefix_caching=True
        # VOLTA GPU: dtype='float32', enable_prefix_caching=True
        """
        args = AsyncEngineArgs(
            model=model_name_or_path,
            dtype=dtype,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True if dtype == 'float32' else False,
            max_model_len=max_model_len,
        )
        self.model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs.from_cli_args(args))

        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            logprobs=logprobs,
            skip_special_tokens=False,
            min_tokens=1,
            max_tokens=max_tokens,
        )
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # there is no actively running loop
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.yes_tokens = None
        self.no_tokens = None
        self.rating_tokens = None

    # TODO: Set the id_tokens as dynamic based on window size
    def set_classification(self, 
        yes_strings=[' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES'],
        no_strings=[' No', 'No', ' no', 'no', 'NO', ' NO'],
        id_strings=[chr(i) for i in range(65, 91)],
        rating_scale: int = 5
    ):
        self.yes_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in yes_strings]
        self.no_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in no_strings]
        self.id_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in id_strings]
        
        # Set up rating tokens for judge scoring (0 to rating_scale)
        self.rating_tokens = {}
        for i in range(rating_scale + 1):
            tokens = [self.tokenizer.encode(f' {i}', add_special_tokens=False)[0],
                     self.tokenizer.encode(f'{i}', add_special_tokens=False)[0]]
            self.rating_tokens[i] = list(set(tokens))
        
        print(f"YES TOKENS: {self.yes_tokens}")
        print(f"NO TOKENS: {self.no_tokens}")
        print(f"ID TOKENS: {self.id_tokens}")
        print(f"RATING TOKENS: {self.rating_tokens}")

    def generate(
        self, 
        prompts, 
        binary_probs: bool = False, 
        dist_logp: bool = False,
        rating_logp: bool = False,
        rating_softmax: bool = False,
        expected_rating: bool = False,
        target_ratings: Optional[List[int]] = None,
        rating_scale: int = 5,
        use_log_scale: bool = False
    ) -> List:
        """
        Generate outputs with various scoring modes.
        
        Args:
            prompts: Input prompts
            binary_probs: Use Yes/(Yes+No) probability scoring
            dist_logp: Use distribution log probability scoring
            rating_logp: Use peak likelihood of target rating(s)
            rating_softmax: Use softmax normalization over target ratings
            expected_rating: Use expected value (weighted sum)
            target_ratings: List of target ratings for logp/softmax modes
            rating_scale: Maximum rating value (default 5)
            use_log_scale: Return log probabilities instead of probabilities
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        return self.loop.run_until_complete(
                self._agenerate(
                    prompts, 
                    use_binary_probs=binary_probs,
                    use_dist_probs=dist_logp,
                    use_rating_logp=rating_logp,
                    use_rating_softmax=rating_softmax,
                    use_expected_rating=expected_rating,
                    target_ratings=target_ratings,
                    rating_scale=rating_scale,
                    use_log_scale=use_log_scale
                )
        )

    async def _iterate_over_output(
        self, 
        output_iterator: AsyncStream, 
        use_binary_probs: bool = False, 
        use_dist_probs: bool = False,
        use_rating_logp: bool = False,
        use_rating_softmax: bool = False,
        use_expected_rating: bool = False,
        target_ratings: Optional[List[int]] = None,
        rating_scale: int = 5,
        use_log_scale: bool = False
    ) -> str:

        async for output in output_iterator:
            if use_binary_probs:
                tok_item = output.outputs[0].logprobs[0]
                yes_ = math.exp(max(
                    [-1e2] + [
                        item.logprob for tok, item in tok_item.items() 
                        if tok in self.yes_tokens
                    ]
                ))
                no_ = math.exp(max(
                    [-1e2] + [
                        item.logprob for tok, item in tok_item.items() 
                        if tok in self.no_tokens 
                    ]
                ))
                output = score = yes_ / (no_ + yes_)

            elif use_rating_logp:
                # Peak likelihood: logP(target_rating) or sum of logP for multiple targets
                tok_item = output.outputs[0].logprobs[0]
                if target_ratings is None:
                    target_ratings = [rating_scale]
                
                target_logps = []
                for rating in target_ratings:
                    if rating in self.rating_tokens:
                        rating_logp = max(
                            [-1e2] + [
                                item.logprob for tok, item in tok_item.items()
                                if tok in self.rating_tokens[rating]
                            ]
                        )
                        target_logps.append(rating_logp)
                
                if target_logps:
                    # Use log-sum-exp for numerical stability when combining multiple targets
                    max_logp = max(target_logps)
                    if use_log_scale:
                        output = max_logp + math.log(sum(math.exp(lp - max_logp) for lp in target_logps))
                    else:
                        output = sum(math.exp(lp) for lp in target_logps)
                else:
                    output = 0.0 if not use_log_scale else -1e2

            elif use_rating_softmax:
                # Softmax normalization over target ratings
                tok_item = output.outputs[0].logprobs[0]
                if target_ratings is None:
                    target_ratings = [rating_scale]
                
                # Get all rating logprobs
                all_logprobs = {}
                for rating in range(rating_scale + 1):
                    if rating in self.rating_tokens:
                        rating_logp = max(
                            [-1e2] + [
                                item.logprob for tok, item in tok_item.items()
                                if tok in self.rating_tokens[rating]
                            ]
                        )
                        all_logprobs[rating] = rating_logp
                
                # Compute softmax over all ratings
                if all_logprobs:
                    max_logp = max(all_logprobs.values())
                    exp_logprobs = {r: math.exp(lp - max_logp) for r, lp in all_logprobs.items()}
                    total = sum(exp_logprobs.values())
                    softmax_probs = {r: exp / total for r, exp in exp_logprobs.items()}
                    
                    # Sum probabilities for target ratings
                    output = sum(softmax_probs.get(r, 0) for r in target_ratings)
                else:
                    output = 0.0

            elif use_expected_rating:
                # Expected rating: sum of P(rating) * rating
                tok_item = output.outputs[0].logprobs[0]
                
                # Get all rating logprobs
                all_logprobs = {}
                for rating in range(rating_scale + 1):
                    if rating in self.rating_tokens:
                        rating_logp = max(
                            [-1e2] + [
                                item.logprob for tok, item in tok_item.items()
                                if tok in self.rating_tokens[rating]
                            ]
                        )
                        all_logprobs[rating] = rating_logp
                
                # Compute softmax and expected value
                if all_logprobs:
                    max_logp = max(all_logprobs.values())
                    exp_logprobs = {r: math.exp(lp - max_logp) for r, lp in all_logprobs.items()}
                    total = sum(exp_logprobs.values())
                    softmax_probs = {r: exp / total for r, exp in exp_logprobs.items()}
                    
                    # Expected value: sum of P(rating) * rating
                    output = sum(prob * rating for rating, prob in softmax_probs.items())
                else:
                    output = 0.0

            # NOTE: the transformation is a bit hacky.
            # NOTE: make sure the numeric identifiers can also work
            elif use_dist_probs:
                tok_item = output.outputs[0].logprobs[0]
                min_logprob = min([item.logprob for item in tok_item.values()])
                output = [min_logprob for _ in self.id_tokens]
                for topk, item in tok_item.items():
                    decoded_token = item.decoded_token.replace('[', '').replace(']', '')
                    if len(decoded_token)==1 and (65 <= ord(decoded_token) <= 90):
                        output[ord(decoded_token)-65] = max(item.logprob, output[ord(decoded_token)-65])
            else:
                output = last_text = output.outputs[0].text
        return output

    async def _agenerate(self, prompts, **kwargs):
        request_ids = [str(uuid.uuid4()) for _ in prompts]

        # Add requests to the engine
        output_iterators = [
            await self.model.add_request(request_id, prompt, self.sampling_params)
            for request_id, prompt in zip(request_ids, prompts)
        ]

        # Gather all the outputs
        outputs = await asyncio.gather(*[
            self._iterate_over_output(output_iterator, **kwargs)
            for output_iterator in output_iterators
        ])
        return list(outputs)
