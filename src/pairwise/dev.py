from prompts.mode import PromptMode, PromptFormatter, Result
from pprint import pprint

# example results
query = "How to improve cardiovascular health?"
passages = [
    "Visiting art galleries fosters creativity and social engagement.",
    "Engaging in at least 150 minutes of moderate aerobic exercise weekly significantly boosts heart health.",
    "Eating a balanced diet rich in fruits, vegetables, and whole grains can improve cardiovascular function.",
    "Regular blood pressure monitoring helps in early detection and prevention of heart disease.",
    "Managing cholesterol levels through lifestyle changes and medication protects arteries from plaque buildup.",
    "Reducing sodium intake can lower blood pressure and reduce heart disease risk.",
    "Getting at least seven hours of quality sleep per night supports overall vascular function.",
    "Reducing alcohol consumption has a positive effect on many aspects of physical wellness.",
    "Spending time outdoors can improve mental health and promote light physical activity.",
    "Learning a musical instrument can help improve cognitive functions in adults.",
]
pairs = []
for i, passage in enumerate(passages):
    pairs.append({'docid': f"docid_{i}", 'score': float(1/ (i+1)), 'content': passage})

results = []
results.append(Result(qid='qid_0', query=query, hits=pairs))

# testing
formatter = PromptFormatter(
    model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
    prompt_mode=PromptMode.RANK_GPT
)

prompts, lengths = formatter.create_prompt_batched(
    results=results,
    rank_start=0,
    rank_end=20
)
