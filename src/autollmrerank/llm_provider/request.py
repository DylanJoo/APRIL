import os
import uuid
import math
import asyncio
import openai
from typing import List
from transformers import AutoTokenizer

class LLM:

    def __init__(
        self,
        api_key: str = 'EMPTY',
        base_url: str = 'http://localhost:8000/v1',
        model_name_or_path: str = 'meta-llama/Llama-3.2-1B-Instruct',
        temperature=0.0,
        top_p=1.0,
        logprobs=20,
        max_tokens=10,
        gpu_memory_utilization=0.9,
        **kwargs
    ):
        print(f"Unused kwargs: {kwargs}")
        self.model_name_or_path = model_name_or_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logprobs = logprobs

        self.client = openai.OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY', api_key),
            base_url=base_url,
            max_retries=10
        )

        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.yes_tokens = None
        self.no_tokens = None
        self.rating_tokens = None
        if logprobs is not None:
            self.set_classification()

    def set_classification(self, 
        yes_strings=[' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES'],
        no_strings=[' No', 'No', ' no', 'no', 'NO', ' NO'],
        id_strings=[chr(i) for i in range(65, 91)],
        rating_strings=[' 0', '0', ' 1', '1', ' 2', '2', ' 3', '3', ' 4', '4', ' 5', '5']
    ):
        self.yes_tokens = [self.tokenizer.tokenize(item)[0] for item in yes_strings]
        self.no_tokens = [self.tokenizer.tokenize(item)[0] for item in no_strings]
        self.id_tokens = [self.tokenizer.tokenize(item)[0] for item in id_strings]

        # also include the strings
        self.yes_tokens += yes_strings
        self.no_tokens += no_strings
        
        # Group rating tokens by their numeric value (0-5)
        self.rating_tokens = {
            i: [self.tokenizer.tokenize(s)[0] for s in rating_strings if s.strip() == str(i)] + 
               [s for s in rating_strings if s.strip() == str(i)]
            for i in range(6)
        }

    def generate(self, prompts, binary_probs=False, dist_logp=False, rating_probs=False) -> List:
        if isinstance(prompts, str):
            prompts = [prompts]
        
        return self.loop.run_until_complete(
                self._agenerate(prompts, 
                                use_binary_probs=binary_probs,
                                use_dist_probs=dist_logp,
                                use_rating_probs=rating_probs)
                )

    async def _agenerate(self, prompts, use_binary_probs=False, use_dist_probs=False, use_rating_probs=False):
        request_ids = [str(uuid.uuid4()) for _ in prompts]

        # Use normal function and add run in thread
        ## NOTE: in serving mode, it will stop util hitting criteria

        def _get_output(prompt, use_binary_probs, use_dist_probs, use_rating_probs):
            response = self.client.completions.create(
                model=self.model_name_or_path,
                prompt=prompt,
                logprobs=self.logprobs,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            if use_binary_probs:
                tok_logps = response.choices[0].logprobs.top_logprobs[0]  # this is strings
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
                output = yes_ / (no_ + yes_)

            elif use_rating_probs:
                tok_logps = response.choices[0].logprobs.top_logprobs[0]  # this is strings
                # Compute probability for each rating (0-5)
                rating_probs = []
                for rating in range(6):
                    rating_logprob = max(
                        [-1e2] + [
                            logp for tok, logp in tok_logps.items()
                            if tok in self.rating_tokens.get(rating, [])
                        ]
                    )
                    rating_probs.append(math.exp(rating_logprob))
                
                # Normalize probabilities
                total_prob = sum(rating_probs)
                if total_prob > 0:
                    rating_probs = [p / total_prob for p in rating_probs]
                
                # Compute expected rating (weighted average)
                output = sum(i * p for i, p in enumerate(rating_probs))

            elif use_dist_probs:
                tok_logps = response.choices[0].logprobs.top_logprobs[0] # this is strings
                min_logprob = min([logp for logp in tok_logps.values()])
                output = [min_logprob for _ in self.id_tokens]
                for topk, logp in tok_logps.items():
                    decoded_token = topk.replace('[', '').replace(']', '')
                    if len(decoded_token)==1 and (65 <= ord(decoded_token) <= 90):
                        output[ord(decoded_token)-65] = max(logp, output[ord(decoded_token)-65])
            else:
                output = response.choices[0].text

            return output

        # Gather all the outputs
        outputs = await asyncio.gather(*[
            asyncio.to_thread(_get_output, prompt,
                use_binary_probs, 
                use_dist_probs,
                use_rating_probs) for prompt in prompts
        ])
        return list(outputs)
