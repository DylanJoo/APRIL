import importlib
from eval_autoqrels import AutoQrel

# Loading
loader = importlib.import_module(f"autollmrerank.loader_dev.irds")
_, _, qrel = loader.load("msmarco-passage/trec-dl-2019/judged")
judge_run = loader.load_run("/users/judylan1/APRIL/runs/Llama-3.3-70B-Instruct/run.msmarco-passage.splade-v3-rerank-rankgpt.trec-dl-2019.txt")
eval_run = loader.load_run("/users/judylan1/APRIL/runs/Llama-3.3-70B-Instruct/run.msmarco-passage.splade-v3-rerank-rankgpt.trec-dl-2019.txt")

autoqrel = AutoQrel(
    qrel=qrel,
    judge_run=judge_run,
    strategies=['direct', 
                'optimal_global', 
                'optimal_per_topic', 
                'optimal_precision', 
                'optimal_recall']
)

for strategy in autoqrel.llm_qrels:
    print(strategy)
    # for qid in ['19335', '47923', '87181', '87452', '104861', '130510', '131843', '146187', '148538', '156493', '168216', '182539', '183378', '207786', '264014', '359349', '405717', '443396', '451602', '489204', '490595', '527433', '573724', '833860', '855410', '915593', '962179', '1037798', '1063750', '1103812', '1106007', '1110199', '1112341', '1113437', '1114646', '1114819', '1115776', '1117099', '1121402', '1121709', '1124210', '1129237', '1133167']:
    for qid in ['19335', '47923']:
        try:
            print(autoqrel.llm_qrels[strategy][qid])
        except:
            print('not found')

