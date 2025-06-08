import math
from transformers import AutoTokenizer
import multiprocessing
import openai
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LLM:

    def __init__(
        self, 
        model="llama3.3-70b-instruct",
        temperature=0.0,
        top_p=1.0,
        max_tokens=10,
        logprobs=True,
        top_logprobs=20
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-70B-Instruct', use_fast=False)
        self.yes_tokens = None
        self.no_tokens = None

    def set_classification(self, yes_strings, no_strings):
        """ Litellm outputs probabilties of each token strings instead of token ids """
        self.yes_tokens = [self.tokenizer.tokenize(item)[0] for item in yes_strings]
        self.no_tokens = [self.tokenizer.tokenize(item)[0] for item in no_strings]

    def preprocess(self, prompts):
        return prompts

    def parallel_call(self, x):
        # preprocess
        x = self.preprocess(x)

        client =  openai.OpenAI(
            api_key=os.environ['OPENAI_API_KEY'], 
            base_url='http://10.162.95.158:4000/v1/'
        )

        # get respoinse
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            messages = [{"role": "user", "content": x}],
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs
        )
        token_top_logprobs = {
             str(item.token): float(item.logprob) for item in response.choices[0].logprobs.content[0].top_logprobs
        }

        # postprocess
        yes_ = math.exp(max( [-1e2] + [logp for tok, logp in token_top_logprobs.items() if tok in self.yes_tokens] ))
        no_ = math.exp(max( [-1e2] + [logp for tok, logp in token_top_logprobs.items() if tok in self.no_tokens] ))
        score = yes_ / (no_ + yes_)
        return score

    def inference(self, x_list, num_processes=32):
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(self.parallel_call, x_list)
        return results
