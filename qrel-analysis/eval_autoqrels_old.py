import argparse
import json
import sys
import os
import importlib
from typing import Optional
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class AutoQrel:

    STRATEGIES = [
        "all", "direct", "thresholding", "rank",
        "largest_gap", "quantile",
        "optimal_per_topic", "optimal_global",
    ]

    def __init__(
        self, 
        qrel, 
        judge_run, 
        strategies,
        threshold=0.5, 
        rank_cutoff=10, 
        gap_k=1,
        quantile_cutoff=0.75, 
        min_relevance=1
    ):
        """
        qrel: dict of dicts {qid: {docid: relevance}}
        judge_run: dict of dicts {qid: {docid: score}} from LLM judge run
        strategies: list of thresholding strategies to apply (or "all" for all strategies)
        threshold: float for binarize strategy
        rank_cutoff: int for rank_cutoff strategy
        gap_k: int for largest_gap strategy
        quantile: float for quantile strategy
        min_relevance: int relevance grade threshold for human qrel
        """
        self.human_qrel = qrel
        if "all" in strategies:
            self.strategies = self.STRATEGIES
            self.strategies.remove('all')
        else:
            self.strategies = strategies

        # Parameters
        self.threshold = threshold
        self.rank_cutoff = rank_cutoff
        self.quantile_cutoff = quantile_cutoff
        self.gap_k = gap_k
        self.min_relevance = min_relevance

        # Precompute LLM-derived qrels for all strategies
        self.llm_qrels = {s: self._dispatch(s, judge_run) for s in self.strategies}

    def _dispatch(self, strategy, run):
        if strategy == "human":
            return self.human_qrel
        if strategy == "direct":
            result = AutoQrel.direct(run)
        elif strategy == "thresholding":
            result = AutoQrel.thresholding(run, self.threshold)
        elif strategy == "rank":
            result = AutoQrel.rank_cutoff(run, self.rank_cutoff)
        elif strategy == "largest_gap":
            result = AutoQrel.largest_gap(run, self.gap_k)
        elif strategy == "quantile":
            result = AutoQrel.quantile(run, self.quantile_cutoff)
        elif strategy == "optimal_global":
            result = AutoQrel.optimal_global(run, self.human_qrel, self.min_relevance)
        elif strategy == "optimal_per_topic":
            result = AutoQrel.optimal_per_topic_f1(run, self.human_qrel)
        elif strategy == "optimal_precision":
            result = AutoQrel.optimal_per_topic_precision(run, self.human_qrel)
        elif strategy == "optimal_recall":
            result = AutoQrel.optimal_per_topic_recall(run, self.human_qrel)
        else:
            raise ValueError(f"Unknown thresholding strategy: {strategy!r}")
        return AutoQrel._filter_no_relevant(result)

    @staticmethod
    def _filter_no_relevant(qrel):
        """Remove qids where all judged documents have zero relevance."""
        return {qid: docs for qid, docs in qrel.items() if any(v > 0 for v in docs.values())}

    @staticmethod
    def direct(run):
        """Round each score to the nearest int → relevance label.
        Best for judge runs whose scores are already relevance levels (e.g. 1.0, 2.0, 3.0)."""
        return {qid: {did: round(score) for did, score in docs.items()} for qid, docs in run.items()}

    @staticmethod
    def thresholding(run, threshold=0.5): # greater and equal to threshold is positive
        return {qid: {did: int(score >= threshold) for did, score in docs.items()}
                for qid, docs in run.items()}

    @staticmethod
    def rank_cutoff(run, topk=10):
        """Top-k ranked docs → 1, remainder → 0.
        Works for any ranked run regardless of score semantics."""
        qrel = {}
        for qid, docs in run.items():
            ranked = list(docs.keys())
            qrel[qid] = {did: (1 if i < topk else 0) for i, did in enumerate(ranked)}
        return qrel

    @staticmethod
    def largest_gap(run, k=1):
        """Binary threshold at the k-th largest consecutive score gap (per topic).

        Scores are sorted descending; consecutive differences are computed. The
        position of the k-th largest gap becomes the positive/negative boundary:
        all docs above (and including) that position are labeled 1, the rest 0.

        k=1 → largest single drop  (most common baseline)
        k=2 → second-largest drop  (useful if the top gap is a tie artefact)
        """
        qrel = {}
        for qid, docs in run.items():
            ranked = sorted(docs.items(), key=lambda x: x[1], reverse=True)
            if len(ranked) < 2:
                qrel[qid] = {did: 1 for did, _ in ranked}
                continue
            scores = [s for _, s in ranked]
            gaps = [(scores[i] - scores[i + 1], i) for i in range(len(scores) - 1)]
            gaps_sorted = sorted(gaps, key=lambda x: x[0], reverse=True)
            cutoff_rank = gaps_sorted[min(k, len(gaps_sorted)) - 1][1]
            qrel[qid] = {
                did: (1 if i <= cutoff_rank else 0)
                for i, (did, _) in enumerate(ranked)
            }
        return qrel

    @staticmethod
    def quantile(run, q=0.75):
        """Per-topic quantile thresholding: score >= q-th percentile → 1, else 0.

        q=0.75 sets the threshold at the 75th percentile of scores for that topic,
        so roughly the top 25% of documents are labeled positive. The quantile is
        computed per-topic so that score scale differences across topics don't bias
        the labeling.
        """
        qrel = {}
        for qid, docs in run.items():
            if not docs:
                qrel[qid] = {}
                continue
            sorted_scores = sorted(docs.values())
            pos = q * (len(sorted_scores) - 1)
            lo, hi = int(pos), min(int(pos) + 1, len(sorted_scores) - 1)
            threshold = sorted_scores[lo] + (pos - lo) * (sorted_scores[hi] - sorted_scores[lo])
            qrel[qid] = {did: (1 if score >= threshold else 0) for did, score in docs.items()}
        return qrel

    @staticmethod
    def _binary_f1(predicted_pos, actual_pos):
        """F1 between two sets of positive doc IDs."""
        tp = len(predicted_pos & actual_pos)
        if tp == 0:
            return 0.0
        precision = tp / len(predicted_pos)
        recall = tp / len(actual_pos)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def optimal_per_topic_f1(run, human_qrel):
        """Oracle: per-topic threshold that maximises F1 against human qrel.

        For every query, all unique scores in the run are tried as candidate
        thresholds; the one yielding the best binary F1 is chosen. This is an
        oracle upper bound — it requires the human labels it is meant to replace.
        """
        qrel = {}
        for qid, docs in run.items():
            hq = human_qrel.get(qid, {})
            actual_pos = {did for did, rel in hq.items() if rel >= 1}
            if not actual_pos:
                qrel[qid] = {did: 0 for did in docs}
                continue
            best_f1, best_threshold = -1.0, None
            for thresh in sorted(set(docs.values()), reverse=True):
                predicted_pos = {did for did, score in docs.items() if score >= thresh}
                f1 = AutoQrel._binary_f1(predicted_pos, actual_pos)
                if f1 > best_f1:
                    best_f1, best_threshold = f1, thresh
            qrel[qid] = {
                did: (1 if score >= best_threshold else 0)
                for did, score in docs.items()
            }
        return qrel

    @staticmethod
    def optimal_per_topic_precision(run, human_qrel):
        qrel = {}
        for qid, docs in run.items():
            hq = human_qrel.get(qid, {})
            actual_pos = {did for did, rel in hq.items() if rel >= 1}
            if not actual_pos:
                qrel[qid] = {did: 0 for did in docs}
                continue
            best_p, best_threshold = -1.0, None
            for thresh in sorted(set(docs.values()), reverse=True):
                predicted_pos = {did for did, score in docs.items() if score >= thresh}
                if not predicted_pos:
                    continue
                top_k = set(sorted(predicted_pos, key=lambda d: docs[d], reverse=True)[:10])
                p = len(top_k & actual_pos) / 10
                if p > best_p:
                    best_p, best_threshold = p, thresh
            qrel[qid] = {
                did: (1 if score >= best_threshold else 0)
                for did, score in docs.items()
            } if best_threshold is not None else {did: 0 for did in docs}
        return qrel

    @staticmethod
    def optimal_per_topic_recall(run, human_qrel):
        qrel = {}
        for qid, docs in run.items():
            hq = human_qrel.get(qid, {})
            actual_pos = {did for did, rel in hq.items() if rel >= 1}
            if not actual_pos:
                qrel[qid] = {did: 0 for did in docs}
                continue
            best_recall, best_threshold = -1.0, None
            for thresh in sorted(set(docs.values()), reverse=True):
                predicted_pos = {did for did, score in docs.items() if score >= thresh}
                recall = len(predicted_pos & actual_pos) / len(actual_pos)
                if recall > best_recall:
                    best_recall, best_threshold = recall, thresh
            qrel[qid] = {
                did: (1 if score >= best_threshold else 0)
                for did, score in docs.items()
            }
        return qrel

    @staticmethod
    def qrel_to_run(qrel, run=None):
        downscaling = 0.01

        # ensure the run and qrel are sorted 
        sorted_qrel = OrderedDict()
        for qid, qrel_hits in qrel.items():
            sorted_qrel[qid] = dict(sorted(qrel_hits.items(), key=lambda x: x[1], reverse=True))

        run = run if run is not None else {}
        result = {}
        for qid, qrel_hits in sorted_qrel.items():
            run_hits = run.get(qid, {})
            run_scores = {did: round(1/i * downscaling, 4) for i, did in \
                    enumerate(run_hits.keys(), start=1)}
            result[qid] = {}
            for docid, score in qrel_hits.items():
                result[qid][docid] = sorted_qrel[qid][docid] + run_scores.get(docid, 0)
        return result

    @staticmethod
    def optimal_global(run, human_qrel, min_relevance=1):
        """Oracle: single global threshold maximising macro-average F1 vs human qrel.

        Every unique score across the entire run is tried as a global threshold;
        the one with the highest average per-topic F1 is selected and applied
        uniformly to all topics.
        """
        all_scores = sorted(
            {score for docs in run.values() for score in docs.values()}, reverse=True
        )
        topics_with_pos = {
            qid for qid, docs in human_qrel.items()
            if any(rel >= min_relevance for rel in docs.values())
        }
        best_avg_f1, best_threshold = -1.0, all_scores[-1]
        for thresh in all_scores:
            f1s = []
            for qid, docs in run.items():
                if qid not in topics_with_pos:
                    continue
                hq = human_qrel.get(qid, {})
                actual_pos = {did for did, rel in hq.items() if rel >= min_relevance}
                predicted_pos = {did for did, score in docs.items() if score >= thresh}
                f1s.append(AutoQrel._binary_f1(predicted_pos, actual_pos))
            avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
            if avg_f1 > best_avg_f1:
                best_avg_f1, best_threshold = avg_f1, thresh
                logger.info(
                    f"Look for optimal threshold." + \
                     "New best threshold={best_threshold:.5g}, avg_F1={best_avg_f1:.4f}"
                )

        return {
            qid: {did: (1 if score >= best_threshold else 0) for did, score in docs.items()}
            for qid, docs in run.items()
        }

if __name__ == "__main__":
    import ir_measures
    from ir_measures import *
    from pprint import pprint

    parser = argparse.ArgumentParser(description="Evaluate retrieval runs against LLM-judge-derived qrel.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--loader_type", type=str, default="irds")
    parser.add_argument("--judge_run", type=str, required=True)
    parser.add_argument("--evaluate_run", type=str, required=True, default=None)

    ## classification thresholding strategies
    parser.add_argument("--strategies", action='append', choices=AutoQrel.STRATEGIES, default=None)

    ### binarize
    parser.add_argument("--threshold", type=float, default=0.5)
    ### rank_cutoff
    parser.add_argument("--rank_cutoff", type=int, default=10, help="Top-k for --thresholding rank_cutoff.")
    parser.add_argument("--gap_k", type=int, default=1, help="k-th largest gap for --thresholding largest_gap.")
    parser.add_argument("--quantile_cutoff", type=float, default=0.75, help="Quantile for --thresholding quantile.")
    parser.add_argument("--min_relevance", type=int, default=1, help="Min relevance grade for oracle strategies.")
    parser.add_argument("--exp", type=str, default=None, help="the experiment tag to record in the output")
    parser.add_argument("--output", type=str, default=None, help="Path to save a per-query CSV for Colab analysis (long format).")
    args = parser.parse_args()

    def strategy_label(strategy, args):
        if strategy == "thresholding":
            return f"thresholding@{args.threshold}"
        elif strategy == "rank":
            return f"rank@{args.rank_cutoff}"
        elif strategy == "largest_gap":
            return f"largest_gap@{args.gap_k}"
        elif strategy == "quantile":
            return f"quantile@{args.quantile_cutoff}"
        return strategy

    # Detect strategies already recorded in the output file so we can skip them
    done_strategies = set()
    if args.output and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    done_strategies.add(json.loads(line.strip())['strategy'])
                except (json.JSONDecodeError, KeyError):
                    pass

    # Loading
    loader = importlib.import_module(f"autollmrerank.loader_dev.{args.loader_type}")
    _, _, qrel = loader.load(args.dataset_name)
    judge_run = loader.load_run(args.judge_run) if args.judge_run else None
    eval_run = loader.load_run(args.evaluate_run)

    # Resolve requested strategies and remove already-done ones
    requested = args.strategies
    strategies_to_run = [s for s in requested if strategy_label(s, args) not in done_strategies]

    autoqrel = AutoQrel(
        qrel=qrel,
        judge_run=judge_run,
        strategies=strategies_to_run,
        threshold=args.threshold,
        rank_cutoff=args.rank_cutoff,
        gap_k=args.gap_k,
        quantile_cutoff=args.quantile_cutoff,
        min_relevance=args.min_relevance,
    )

    judge_name = os.path.basename(args.judge_run)
    results = []

    # Evaluate the judge run against the human qrel (only if not already recorded)
    if 'human' not in done_strategies:
        r = ir_measures.calc_aggregate([nDCG@10], qrel, judge_run)[nDCG@10]
        results.append({
            'dataset': args.dataset_name,
            'exp': args.exp,
            'judge_run': 'human.qrel',
            'evaluate_run': judge_name,
            'strategy': 'human',
            'nDCG@10': round(r, 4),
        })

    # Evaluate against each thresholding strategy
    eval_name = os.path.basename(args.evaluate_run)
    for strategy, llm_qrel in autoqrel.llm_qrels.items():
        llm_qrel = {qid: item for qid, item in llm_qrel.items() if qid in eval_run}
        r = ir_measures.calc_aggregate([nDCG@10], llm_qrel, eval_run)[nDCG@10]
        results.append({
            'dataset': args.dataset_name,
            'exp': args.exp,
            'judge_run': judge_name,
            'evaluate_run': eval_name,
            'strategy': strategy,
            'nDCG@10': round(r, 4),
        })

    if not results:
        logger.info("All strategies already computed for this output file; nothing to do.")
        sys.exit(0)

    # Append to existing file if we skipped some strategies, otherwise write fresh
    if args.output:
        mode = 'a' if done_strategies else 'w'
        with open(args.output, mode) as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
