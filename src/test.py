from vllm import LLM, SamplingParams
from utils.tools import batch_iterator

# model, sampling_params
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, prompt_logprobs=True, max_tokens=1)
model = LLM(model="facebook/opt-125m")
tokenizer = model.get_tokenizer()

# prompt preparation
template = "Please write a question based on this passage.\nPassage: This is testing document.\nQuestion: This is a testing query."

all_tokens = tokenizer(template)
prompt_tokens = tokenizer(' This is a testing query.', add_special_tokens=False)['input_ids']

id_pairs, prompts = [], []
prompts.append( template )

# inference
batch = prompts * 5
outputs = model.generate(batch, sampling_params)
for output in outputs:
    logprobs = [output.prompt_logprobs[i][id].logprob if i > 0 else -1 for i, id in enumerate(output.prompt_token_ids)][-len(prompt_tokens):]
    print(tokenizer.tokenize(' This is a testing query.'))
    print(prompt_tokens)
    print(logprobs)
