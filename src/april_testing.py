from pairwise.rank_llm.rankllm import PromptMode, RankPairwiseOSLLM
from pairwise.rank_llm.utils import Result

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

agent = RankPairwiseOSLLM(
    model='llama3.3-70b-instruct',
    context_size=10240,
    prompt_mode=PromptMode.APRIL,
    num_few_shot_examples=0,
    device="cuda",
    num_gpus=1,
    variable_passages=False,
    window_size=20,
    system_message="You are CompareLLM, an intelligent assistant that can analyze list of passages and compare the two assigned passages based on their relevancy to the query.",
    batched=True,
)

## ps
results = agent.sliding_windows_batched(
    results,
    use_logits=False, # set to True if comparing label's probabilities
    use_alpha=False,
    rank_start=0,
    rank_end=20,
    window_size=20,
    step=10,
)
print(results)
