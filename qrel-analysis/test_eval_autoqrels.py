import ir_measures
from ir_measures import *
import importlib
from eval_autoqrels import AutoQrel

# Loading
loader = importlib.import_module(f"autollmrerank.loader_dev.irds")
_, _, qrel = loader.load("msmarco-passage/trec-dl-2019/judged")
judge_run = loader.load_run("/users/judylan1/APRIL/runs/Llama-3.3-70B-Instruct/run.msmarco-passage.bm25-rerank-setmaxheaptopk.trec-dl-2019.txt")
eval_run = loader.load_run("/users/judylan1/APRIL/runs/Llama-3.3-70B-Instruct/run.msmarco-passage.bm25-rerank-setmaxheaptopk.trec-dl-2019.txt")

autoqrel = AutoQrel(
    qrel=qrel,
    judge_run=judge_run,
    strategies=['direct', 
                'optimal_global', 
                'optimal_per_topic', 
                'optimal_precision', 
                'optimal_recall']
)

for strategy, llm_qrel in autoqrel.llm_qrels.items():
    llm_qrel = {qid: item for qid, item in llm_qrel.items() if qid in eval_run}
    r = ir_measures.calc_aggregate([nDCG@10], llm_qrel, eval_run)[nDCG@10]
    print(strategy, r)
