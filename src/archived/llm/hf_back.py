import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import time
import json
import string
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

MODEL_CLASS = {
    "clm": AutoModelForCausalLM
}

class LLM:

    def __init__(self, 
        model,
        model_class='CLM',
        temperature=0.7, 
        top_p=1.0, 
        flash_attention_2=True,
        device='auto'
    ):
        start_time = time.time()

        if flash_attention_2:
            model_kwargs = {
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16
            }
        else:
            model_kwargs = {'torch_dtype': torch.float16}

        self.model = MODEL_CLASS[model_class.lower()].from_pretrained(
            model,
            device_map=device,
            **model_kwargs
        )
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token # during trainin, no padding is needed

        self.yes_tokens = None
        self.no_tokens = None
        logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    def set_classification(self, yes_tokens, no_tokens):
        """ HF default generated output logits can be traced by token ids """
        self.yes_tokens = [self.tokenizer.tokenize(item)[0] for item in yes_tokens]
        self.no_tokens = [self.tokenizer.tokenize(item)[0] for item in no_tokens]

    def preprocess(self, x):
        return x

    def inference(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]

        # generate
        _, batch_logits = self.generate(prompts, min_tokens=1, max_tokens=3)

        # collect scores 
        scores = []
        for logits in batch_logits: # (B, L, N)
            yes_ = math.exp(max( [logits[-1, tok_id] for tok_id in self.yes_tokens] ))
            no_ = math.exp(max( [logits[-1, tok_id] for tok_id in self.no_tokens] ))
            scores.append( (yes_) / (no_ + yes_) )
        return scores

    def generate(self, x, min_tokens=0, max_tokens=1024, **kwargs):

        x = self.preprocess(x)
        inputs = self.tokenizer(x, padding=True, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature, 
            top_p=self.top_p, 
            min_new_tokens=min_tokens,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            return_dict_in_generate=True, 
            output_logits=True
        )
        
        generation = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        output_logits = torch.stack(outputs.logits, dim=1)

        del inputs, outputs
        torch.cuda.empty_cache()

        return generation, output_logits
