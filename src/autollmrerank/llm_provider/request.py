import os
import uuid
import math
import asyncio
import openai
from typing import List, Optional
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
        rating_scale: int = 5
    ):
        self.yes_tokens = [self.tokenizer.tokenize(item)[0] for item in yes_strings]
        self.no_tokens = [self.tokenizer.tokenize(item)[0] for item in no_strings]
        self.id_tokens = [self.tokenizer.tokenize(item)[0] for item in id_strings]

        # also include the strings
        self.yes_tokens += yes_strings
        self.no_tokens += no_strings

        # Set up rating tokens for judge scoring (0 to rating_scale)
        self.rating_tokens = {}
        for i in range(rating_scale + 1):
            tokens = [self.tokenizer.tokenize(f' {i}')[0] if self.tokenizer.tokenize(f' {i}') else f' {i}',
                     self.tokenizer.tokenize(f'{i}')[0] if self.tokenizer.tokenize(f'{i}') else f'{i}']
            # Include raw strings as fallback
            tokens.extend([f' {i}', f'{i}'])
            self.rating_tokens[i] = list(set(tokens))

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

    async def _agenerate(
        self, 
        prompts, 
        use_binary_probs: bool = False, 
        use_dist_probs: bool = False,
        use_rating_logp: bool = False,
        use_rating_softmax: bool = False,
        use_expected_rating: bool = False,
        target_ratings: Optional[List[int]] = None,
        rating_scale: int = 5,
        use_log_scale: bool = False
    ):
        request_ids = [str(uuid.uuid4()) for _ in prompts]

        # Use normal function and add run in thread
        ## NOTE: in serving mode, it will stop util hitting criteria

        def _get_output(prompt, use_binary_probs, use_dist_probs, use_rating_logp, 
                       use_rating_softmax, use_expected_rating, target_ratings,
                       rating_scale, use_log_scale):
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

            elif use_rating_logp:
                # Peak likelihood: logP(target_rating)
                tok_logps = response.choices[0].logprobs.top_logprobs[0]
                if target_ratings is None:
                    target_ratings = [rating_scale]
                
                target_logps = []
                for rating in target_ratings:
                    if rating in self.rating_tokens:
                        rating_logp_val = max(
                            [-1e2] + [
                                logp for tok, logp in tok_logps.items()
                                if tok in self.rating_tokens[rating]
                            ]
                        )
                        target_logps.append(rating_logp_val)
                
                if target_logps:
                    max_logp = max(target_logps)
                    if use_log_scale:
                        output = max_logp + math.log(sum(math.exp(lp - max_logp) for lp in target_logps))
                    else:
                        output = sum(math.exp(lp) for lp in target_logps)
                else:
                    output = 0.0 if not use_log_scale else -1e2

            elif use_rating_softmax:
                # Softmax normalization over target ratings
                tok_logps = response.choices[0].logprobs.top_logprobs[0]
                if target_ratings is None:
                    target_ratings = [rating_scale]
                
                all_logprobs = {}
                for rating in range(rating_scale + 1):
                    if rating in self.rating_tokens:
                        rating_logp_val = max(
                            [-1e2] + [
                                logp for tok, logp in tok_logps.items()
                                if tok in self.rating_tokens[rating]
                            ]
                        )
                        all_logprobs[rating] = rating_logp_val
                
                if all_logprobs:
                    max_logp = max(all_logprobs.values())
                    exp_logprobs = {r: math.exp(lp - max_logp) for r, lp in all_logprobs.items()}
                    total = sum(exp_logprobs.values())
                    softmax_probs = {r: exp / total for r, exp in exp_logprobs.items()}
                    output = sum(softmax_probs.get(r, 0) for r in target_ratings)
                else:
                    output = 0.0

            elif use_expected_rating:
                # Expected rating: sum of P(rating) * rating
                tok_logps = response.choices[0].logprobs.top_logprobs[0]
                
                all_logprobs = {}
                for rating in range(rating_scale + 1):
                    if rating in self.rating_tokens:
                        rating_logp_val = max(
                            [-1e2] + [
                                logp for tok, logp in tok_logps.items()
                                if tok in self.rating_tokens[rating]
                            ]
                        )
                        all_logprobs[rating] = rating_logp_val
                
                if all_logprobs:
                    max_logp = max(all_logprobs.values())
                    exp_logprobs = {r: math.exp(lp - max_logp) for r, lp in all_logprobs.items()}
                    total = sum(exp_logprobs.values())
                    softmax_probs = {r: exp / total for r, exp in exp_logprobs.items()}
                    output = sum(prob * rating for rating, prob in softmax_probs.items())
                else:
                    output = 0.0

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
                use_rating_logp,
                use_rating_softmax,
                use_expected_rating,
                target_ratings,
                rating_scale,
                use_log_scale) for prompt in prompts
        ])
        return list(outputs)
