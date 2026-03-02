"""
Hard-negative classification wrapper.

Instead of reranking all top-K candidates, this wrapper constructs a small
classification pool per query consisting of:
  - positive document(s) from qrels (relevance >= pos_threshold)
  - hard negatives sampled from the first-stage retrieval run

The LLM then scores each document in the pool. This is substantially cheaper
than full reranking for datasets with many queries, because the pool is small
(e.g. 1 positive + 9 hard negatives = 10 docs vs. top-100).
"""

import os
from typing import Optional, Dict, List
from pprint import pprint
from tqdm import tqdm
import torch
from functools import wraps
import time
import random

from .utils import Result, batch_iterator
from .input_assembler import AutoAssembler
from .prompt_builder import PromptBuilder
from .result_parser import ResultParser
from .config_manager import ConfigManager


class AutoLLMClassifier:
    """
    Hard-negative classification evaluator.

    Builds a per-query pool of (positives + hard negatives) and scores them
    with an LLM judge, avoiding the cost of reranking all top-K candidates.
    """

    @classmethod
    def from_prebuilt(cls, method_name, model_name_or_path, **kwargs) -> "AutoLLMClassifier":
        import importlib.resources as pkg_resources
        path = pkg_resources.files("autollmrerank.configs").joinpath(f"{method_name}.yaml")
        llmconfig = {'model_name_or_path': model_name_or_path}
        llmconfig.update(kwargs.pop('llm', {}))
        config = ConfigManager(path=path, llm=llmconfig, **kwargs).get_config()
        return cls(config, **kwargs)

    @staticmethod
    def timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"\n\n{func.__qualname__} took {end - start:.6f} seconds")
            return result
        return wrapper

    def __init__(self, config, **kwargs) -> None:
        self.config = config
        prompt_builder = PromptBuilder(config=config)

        if config.llm.backend == 'vllm':
            from .llm_provider.vllm_dev import LLM
        if (config.llm.backend == 'openai') or (config.llm.backend == 'request'):
            from .llm_provider.request import LLM
        if config.llm.backend == 'vllm_dev':
            from .llm_provider.vllm_dev import LLM

        agent = LLM(
            model_name_or_path=config.llm.model_name_or_path,
            temperature=config.llm.temperature,
            top_p=config.llm.top_p,
            logprobs=20 if config.llm.use_logits else None,
            max_model_len=config.llm.max_model_len,
            max_tokens=5 if config.llm.use_logits else 128,
            dtype=config.llm.dtype,
            num_gpus=max(1, int(torch.cuda.device_count())),
            base_url=('http://localhost:8000/v1' or config.llm.base_url),
            api_key='EMPTY'
        )

        result_parser = ResultParser(use_alpha=config.use_alphabetical)

        self.assembler = AutoAssembler.from_config(
            config,
            prompt_builder=prompt_builder,
            llm_provider=agent,
            result_parser=result_parser,
        )

    @staticmethod
    def build_classification_pool(
        run: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        num_hard_negatives: int = 25,
        neg_sampling: str = 'depth-n',
        seed: int = 42,
    ) -> List[Result]:
        """
        Construct a classification pool per query.

        Args:
            run: First-stage retrieval results {qid: {docid: score}}.
            qrels: Relevance judgments {qid: {docid: rel}}.
            corpus: Document content {docid: content_dict}.
            queries: Query texts {qid: query_text}.
            pos_threshold: Minimum qrel relevance to be considered a positive.
            num_hard_negatives: Number of hard negatives per query.
            neg_sampling: Strategy for selecting hard negatives from the run:
                - 'top': top-ranked non-relevant docs (hardest negatives)
                - 'random': randomly sampled non-relevant docs
                - 'mixed': half top-ranked, half randomly sampled
            max_positives: If set, cap the number of positives per query.
                           None = use all positives found in qrels.
            seed: Random seed for reproducible negative sampling.

        Returns:
            List of Result objects, one per query, with the classification pool as hits.
        """
        rng = random.Random(seed)
        results = []
        skipped = 0
        pos_threshold = 1

        for qid in qrels:

            # Collect positives from qrels that exist in corpus
            pos_docids = [docid for docid, rel in qrels[qid].items() if rel >= pos_threshold]
            neg_docids = [docid for docid, rel in qrels[qid].items() if rel < pos_threshold]

            # Collect hard negatives: ranked docs not in qrels at all
            # (exclude all judged docs to avoid using unjudged docs as negatives)
            neg_candidates = []

            if neg_sampling == 'depth-n':
                n = num_hard_negatives // len(run)  # per-run quota
                for run in runs:
                    neg_candidates.extend([docid for docid in run.get(qid, {}) if docid not in qrels[qid]][:n])
                neg_docids = neg_candidates

            if neg_sampling == 'random':
                for run in runs:
                    neg_candidates.extend([docid for docid in run.get(qid, {}) if docid not in qrels[qid]])
                neg_docids = rng.sample(neg_candidates, min(num_hard_negatives, len(neg_candidates)))

            if neg_sampling == 'fusion':
                smooth_const = 60
                docid_scores = {}
                for run in runs:
                    for rank, docid in enumerate(run.get(qid, {}), start=1):
                        if docid not in qrels[qid]:
                            docid_scores[docid] = docid_scores.get(docid, 0) + 1 / (rank + smooth_const)

                sorted_negatives = sorted(docid_scores.items(), key=lambda x: x[1], reverse=True)
                neg_docids = [docid for docid, score in sorted_negatives[:num_hard_negatives]]

            # Build pool and shuffle to avoid position bias
            hits = []
            for docid in pos_docids + neg_docids:
                hits.append({
                    'docid': docid,
                    'score': float(query_run.get(docid, 0.0)),
                    'content_dict': corpus[docid],
                    'is_positive': docid in pos_set,  # metadata for offline analysis
                })

            results.append(Result(qid=qid, query=queries[qid], hits=hits))

        return results

    @timer
    def classify(
        self,
        run: Dict[str, Dict[str, float]],
        queries: Dict[str, str],
        corpus: Dict[str, Dict[str, str]],
        qrels: Dict[str, Dict[str, int]],
        pos_threshold: int = 1,
        num_hard_negatives: int = 9,
        neg_sampling: str = 'top',
        max_positives: Optional[int] = None,
        query_batch_size: int = 32,
        seed: int = 42,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run hard-negative classification scoring.

        Args:
            run: First-stage retrieval {qid: {docid: score}}.
            queries: Query texts {qid: query_text}.
            corpus: Document corpus {docid: content_dict}.
            qrels: Relevance judgments {qid: {docid: rel}}.
            pos_threshold: Minimum qrel relevance to count as positive.
            num_hard_negatives: Number of hard negatives per query.
            neg_sampling: Negative sampling strategy ('top', 'random', 'mixed').
            max_positives: Cap positives per query (None = use all).
            query_batch_size: Number of queries processed per LLM batch.
            seed: RNG seed for reproducibility.

        Returns:
            Scored run {qid: {docid: score}} over the classification pool only.
        """
        init_results = self.build_classification_pool(
            run=run,
            qrels=qrels,
            corpus=corpus,
            queries=queries,
            strategy=strategym
        )

        scored_results = []
        for batch_results in tqdm(
            batch_iterator(init_results, size=query_batch_size),
            desc=f"Scoring classification pools (batch={query_batch_size})",
            total=len(init_results) // query_batch_size + 1,
        ):
            # Pool sizes can vary across queries; use the max in this batch
            rank_end = max(len(r.hits) for r in batch_results)
            batch_scored = self.assembler.run(
                init_results=batch_results,
                rank_start=0,
                rank_end=rank_end,
                batch_size=query_batch_size,
                num_runs=getattr(self.config, 'num_runs', 1),
            )
            scored_results.extend(batch_scored)

        for r in scored_results:
            r.sort_by(field='score')

        # Convert back to run format (pool docs only)
        scored_run = {}
        for result in scored_results:
            scored_run[result.qid] = {}
            for rank, hit in enumerate(result.hits, start=1):
                scored_run[result.qid][hit['docid']] = hit.get('score', 1.0 / rank)

        return scored_run


if __name__ == "__main__":
    import ir_measures
    from ir_measures import *
    import importlib

    config = ConfigManager().get_config()
    config_dict = ConfigManager().get_config(return_dict=True)
    pprint(config_dict)

    # Init classifier (uses same assembler/LLM infrastructure as wrapper.py)
    classifier = AutoLLMClassifier(config)

    # Load data
    loader = importlib.import_module(
        f"autollmrerank.loader_dev.{config.data.loader_type}", package=__name__
    )
    run = loader.load_run(config.data.input_run)
    corpus, queries, qrels = loader.load(config.data.dataset_name, query_fields=None, doc_fields=None)
    run = {qid: hits for qid, hits in run.items() if qid in qrels}

    # Classification-specific config (falls back to sensible defaults)
    pos_threshold      = getattr(config.data, 'pos_threshold', 1)
    num_hard_negatives = getattr(config.data, 'num_hard_negatives', 9)
    neg_sampling       = getattr(config.data, 'neg_sampling', 'top')
    max_positives      = getattr(config.data, 'max_positives', None)

    # Run classification
    scored_run = classifier.classify(
        run=run,
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        pos_threshold=pos_threshold,
        num_hard_negatives=num_hard_negatives,
        neg_sampling=neg_sampling,
        max_positives=max_positives,
        query_batch_size=config.data.batch_size,
    )

    # Output scored run
    if config.data.output_run is None:
        output_path = config.data.input_run.replace(
            'runs', f'runs/{config.rerank_mode}/classify'
        )
    else:
        output_path = config.data.output_run
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for qid in scored_run:
            for i, (docid, score) in enumerate(scored_run[qid].items()):
                f.write(f"{qid} Q0 {docid} {i+1} {score} {config.rerank_mode}\n")

    # Evaluate on the classification pool
    # Note: scores only cover pool docs, so this measures discrimination within the pool
    r1 = ir_measures.calc_aggregate([nDCG@10, MRR@10, P@1], qrels, run)
    r2 = ir_measures.calc_aggregate([nDCG@10, MRR@10, P@1], qrels, scored_run)

    eval_log = {
        'rerank_mode':         config.rerank_mode,
        'model_name_or_path':  config.llm.model_name_or_path,
        'dataset_name':        f"{config.data.loader_type}:{config.data.dataset_name}",
        'run_path':            config.data.input_run,
        'pos_threshold':       pos_threshold,
        'num_hard_negatives':  num_hard_negatives,
        'neg_sampling':        neg_sampling,
        'retrieval (pool)':    r1,
        'classified':          r2,
    }
    pprint(eval_log)
