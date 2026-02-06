import uuid
import math
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
from typing import List
from contextlib import contextmanager
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
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if logprobs:
            self.set_classification()

    # TODO: Set the id_tokens as dynamic based on window size
    def set_classification(self, 
        yes_strings=[' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES'],
        no_strings=[' No', 'No', ' no', 'no', 'NO', ' NO'],
        id_strings=[chr(i) for i in range(65, 91)],
        max_rating=5, 
        target_ratings=[3,4,5],
    ):
        # self.yes_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in yes_strings]
        # self.no_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in no_strings]
        self.id_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in id_strings]
        self.max_rating = max_rating
        self.target_ratings = target_ratings
        # print(f"YES TOKENS: {self.yes_tokens} | NO TOKENS: {self.no_tokens}")
        print(f"ID TOKENS: {self.id_tokens}")
        print(f"MAX RATING TOKEN: {self.max_rating} | TARGET TOKENS: {target_ratings}")

    def generate(
        self, 
        prompts, 
        binary_probs=False, 
        dist_logp=False,
        rating_logp=False,
        expected_rating=False,
    ) -> List:
        if isinstance(prompts, str):
            prompts = [prompts]

        return self.loop.run_until_complete(
                self._agenerate(prompts, 
                                use_binary_probs=binary_probs,
                                use_dist_probs=dist_logp,
                                use_rating_logp=rating_logp,
                                use_expected_rating=expected_rating)
        )

    async def _iterate_over_output(
        self, 
        output_collector, 
        use_binary_probs=False, 
        use_dist_probs=False,
        use_rating_logp=False,
        use_expected_rating=False,
    ) -> str:

        self.model._run_output_handler()
        finished = False
        while not finished:

            response = output_collector.get_nowait() or await output_collector.get()
            finished = response.finished

            if use_binary_probs:
                # TODO: make it more flexible. Instead of defining the yes no in the begining
                tok_item = response.outputs[0].logprobs[0]
                yes_ = math.exp(max(
                    [-1e2] + [
                        item.logprob for tok, item in tok_item.items() 
                        if 'yes' in item.decoded_token.lower()
                    ]
                ))
                no_ = math.exp(max(
                    [-1e2] + [
                        item.logprob for tok, item in tok_item.items() 
                        if 'no' in item.decoded_token.lower()
                    ]
                ))
                result = yes_ / (no_ + yes_)

            elif (use_rating_logp) or (use_expected_rating):
                tok_item = response.outputs[0].logprobs[0]
                rating_logp = [0.0 for _ in range(self.max_rating+1)]
                print('output tokens', [item.decoded_token for item in tok_item.values()])
                for topk, item in tok_item.items():
                    for r in range(self.max_rating+1):
                        if str(r) in item.decoded_token:
                            rating_logp[r] = max(math.exp(item.logprob), rating_logp[r])

                if use_expected_rating:
                    result = sum([r * rating_logp[r] for r in self.target_ratings]) / sum(rating_logp)
                    print('exp rating', result)
                else:
                    target_ = sum([rating_logp[r] for r in self.target_ratings])
                    result = target_ / sum(rating_logp)

            # NOTE: the transformation is a bit hacky.
            # NOTE: make sure the numeric identifiers can also work
            elif use_dist_probs:
                tok_item = response.outputs[0].logprobs[0]
                min_logprob = min([item.logprob for item in tok_item.values()])
                result = [min_logprob for _ in self.id_tokens]
                for topk, item in tok_item.items():
                    decoded_token = item.decoded_token.replace('[', '').replace(']', '')
                    if len(decoded_token)==1 and (65 <= ord(decoded_token) <= 90):
                        result[ord(decoded_token)-65] = max(item.logprob, result[ord(decoded_token)-65])
            else:
                result = response.outputs[0].text

        return result

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

    @contextmanager
    def default(self):
        """
        Usage example:
        with llm.default():
            outputs = llm.generate(prompts)
        """
        old_sampling_params = self.sampling_params
        try:
            print("Entering default sampling parameters context ...\nTemperature: 1.0, Top-p: 1.0, Max tokens: 512")
            self.sampling_params.temperature = 1.0
            self.sampling_params.top_p = 1.0
            self.sampling_params.max_tokens = 512
            yield  # This is where the code inside the 'with' block runs
        finally:
            self.sampling_params = old_sampling_params
