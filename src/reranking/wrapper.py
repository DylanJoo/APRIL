""" [NOTE] add configuration control. """
from typing import Optional, Tuple, List, Dict, Union, Any
from reranking.utils import RerankMode, Result
from reranking.input_assembler import BubbleSort
from reranking.config_manager import ConfigManager
from pprint import pprint
# import ir_measures
# from ir_measures import *
# import os
# import loader
# home_dir=str(Path.home())

# class ModularReranker:
#
#     def __init__(self, config, **kwargs) -> None:
#         self.assembler = BubbleSort(config)
#
#     @staticmethod
#     def convert_run_to_result(run, queries=None, corpus=None):
#         results = []
#         for qid, hits in run.items():
#             query = queries[qid]
#             hits = []
#             for docid, score in hits.items():
#                 hits.append({'docid': docid, 'score': float(score), 'content': corpus[docid]['contents']})
#             results.append(Result(qid=qid, query=query, hits=hits))
#         return results
#
#     def rerank(
#         self,
#         run: Dict[str, Dict[str, float]],
#         queries: Dict[str, str],
#         corpus: Dict[str, Dict[str, str]],
#         batch_size: int = 64,
#     ) -> Dict[str, Dict[str, float]]:
#         """
#         Args
#             run (Dict[str, Dict[str, float]]): The initial run to be reranked.
#             queries: (Dict[str, str]): A dictionary mapping query IDs to query strings.
#             corpus (Dict[str, Dict[str, str]]): A dictionary mapping document IDs to their content and title (if applicable).
#         """
#         init_results = self.convert_run_to_result(run, queries, corpus)
#
#         reranked_results = self.assembler.run(
#             retrieved_run=init_results,
#             batch_size=batch_size,
#         )
#
#         return 0

if __name__ == '__main__':
    ALPH_START_IDX = ord('A')-1

    cfg = ConfigManager("reranking/configs/default_config.yaml").get_config()
    pprint(cfg)

    # modurlar_reranker = ModularReranker(
    #     model_name_or_path=args.model_name_or_path,
    #     rerank_mode=args.rerank_mode,
    #     include_system_message=args.include_system_message,
    #     system_message=args.system_message,
    #     context_size=args.context_size,
    #     window_size=args.window_size,
    #     step_size=args.step_size,
    #     batched=args.batched,
    #     backend=args.backend
    # )
    #
    # modular_reranker.rerank(
    #     run={},
    #     queries={},
    #     corpus={},
    #     batch_size=64
    # )
