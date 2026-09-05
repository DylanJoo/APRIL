"""
qrel_stats.py

Compute and compare statistics for TREC-format qrel files.

TREC qrel format: <qid> <iter> <docid> <relevance>

Usage
-----
# Single file
python qrel_stats.py path/to/autollmqrel.rank@10.txt

# Multiple files
python qrel_stats.py autollmqrel.rank@10.txt autollmqrel.thresholding@0.5.txt

# All files in a directory
python qrel_stats.py --dir autoqrels/pool-40-systems-top10-rerank-judge/trec-dl-2019/

# Compare against a reference (human) qrel
python qrel_stats.py --ref human.qrel autollmqrel.rank@10.txt autollmqrel.thresholding@0.5.txt

# Output as JSON
python qrel_stats.py --json autollmqrel.rank@10.txt
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# {qid: {docid: relevance}}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_qrel(path):
    qrel = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _iter, docid, rel = parts[0], parts[1], parts[2], parts[3]
            qrel[qid][docid] = int(rel)
    return dict(qrel)


# ---------------------------------------------------------------------------
# Statistics dataclass
# ---------------------------------------------------------------------------

@dataclass
class QrelStats:
    name: str = ""
    num_queries: int = 0
    num_judged: int = 0
    num_relevant: int = 0
    num_nonrelevant: int = 0
    rel_ratio: float = 0.0
    rel_distribution: Dict[str, int] = field(default_factory=dict)
    avg_judged_per_query: float = 0.0
    min_judged_per_query: int = 0
    max_judged_per_query: int = 0
    avg_relevant_per_query: float = 0.0
    min_relevant_per_query: int = 0
    max_relevant_per_query: int = 0
    queries_no_relevant: int = 0

    # Populated only when a reference qrel is given
    ref_name: str = ""
    shared_queries: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    agreement: Optional[float] = None   # fraction of shared (q,d) pairs with same label


def compute_stats(qrel, name=""):
    s = QrelStats(name=name)
    s.num_queries = len(qrel)

    judged_counts = []
    relevant_counts = []
    rel_dist = defaultdict(int)

    for docs in qrel.values():
        n_judged = len(docs)
        n_rel = sum(1 for r in docs.values() if r > 0)
        judged_counts.append(n_judged)
        relevant_counts.append(n_rel)
        for r in docs.values():
            rel_dist[r] += 1

    s.num_judged = sum(judged_counts)
    s.num_relevant = sum(relevant_counts)
    s.num_nonrelevant = s.num_judged - s.num_relevant
    s.rel_ratio = s.num_relevant / s.num_judged if s.num_judged else 0.0
    s.rel_distribution = {str(k): v for k, v in sorted(rel_dist.items())}

    if judged_counts:
        s.avg_judged_per_query = s.num_judged / s.num_queries
        s.min_judged_per_query = min(judged_counts)
        s.max_judged_per_query = max(judged_counts)
        s.avg_relevant_per_query = s.num_relevant / s.num_queries
        s.min_relevant_per_query = min(relevant_counts)
        s.max_relevant_per_query = max(relevant_counts)
        s.queries_no_relevant = sum(1 for c in relevant_counts if c == 0)

    return s


def add_comparison(stats, qrel, ref, ref_name=""):
    """Augment stats in-place with precision/recall/F1/agreement vs ref."""
    stats.ref_name = ref_name
    shared_qids = set(qrel) & set(ref)
    stats.shared_queries = len(shared_qids)

    tp = fp = fn = 0
    agree = total_shared_pairs = 0

    for qid in shared_qids:
        pred_pos = {did for did, r in qrel[qid].items() if r > 0}
        true_pos = {did for did, r in ref[qid].items() if r > 0}
        tp += len(pred_pos & true_pos)
        fp += len(pred_pos - true_pos)
        fn += len(true_pos - pred_pos)

        shared_pairs = set(qrel[qid]) & set(ref[qid])
        agree += sum(1 for did in shared_pairs if qrel[qid][did] == ref[qid][did])
        total_shared_pairs += len(shared_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    stats.precision = round(precision, 4)
    stats.recall = round(recall, 4)
    stats.f1 = round(f1, 4)
    stats.agreement = round(agree / total_shared_pairs, 4) if total_shared_pairs else None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(v, decimals=2):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return "{:.{}f}".format(v, decimals)
    return str(v)


def print_stats_table(all_stats, show_comparison):
    col_w = max(30, max(len(s.name) for s in all_stats) + 2)

    def row(label, *vals):
        print("  {:<32}".format(label) + "".join("{:>{}}".format(v, col_w) for v in vals))

    header = [s.name for s in all_stats]
    width = 34 + col_w * len(all_stats)
    print("  {:<32}".format("Metric") + "".join("{:>{}}".format(h, col_w) for h in header))

    row("Queries",              *[s.num_queries for s in all_stats])
    row("Judged pairs",         *[s.num_judged for s in all_stats])
    row("Relevant pairs",       *[s.num_relevant for s in all_stats])
    row("Non-relevant pairs",   *[s.num_nonrelevant for s in all_stats])
    row("Relevance ratio",      *[_fmt(s.rel_ratio, 3) for s in all_stats])
    row("Queries w/o relevant", *[s.queries_no_relevant for s in all_stats])
    row("Avg judged/query",     *[_fmt(s.avg_judged_per_query) for s in all_stats])
    row("Min/Max judged/query", *["{}/{}".format(s.min_judged_per_query, s.max_judged_per_query) for s in all_stats])
    row("Avg relevant/query",   *[_fmt(s.avg_relevant_per_query) for s in all_stats])
    row("Min/Max relevant/q",   *["{}/{}".format(s.min_relevant_per_query, s.max_relevant_per_query) for s in all_stats])

    all_grades = sorted({g for s in all_stats for g in s.rel_distribution}, key=int)
    for g in all_grades:
        row("  rel={} count".format(g), *[s.rel_distribution.get(g, 0) for s in all_stats])

    if show_comparison:
        print("-" * width)
        row("Reference",        *[s.ref_name or "N/A" for s in all_stats])
        row("Shared queries",   *[_fmt(s.shared_queries, 0) for s in all_stats])
        row("Precision",        *[_fmt(s.precision, 4) for s in all_stats])
        row("Recall",           *[_fmt(s.recall, 4) for s in all_stats])
        row("F1",               *[_fmt(s.f1, 4) for s in all_stats])
        row("Agreement",        *[_fmt(s.agreement, 4) for s in all_stats])

    print("=" * width)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics for TREC-format qrel files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("qrels", nargs="*", help="One or more qrel files.")
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory — load all *.txt files inside.")
    parser.add_argument("--ref", type=str, default=None,
                        help="Reference (human) qrel for comparison.")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of a table.")
    args = parser.parse_args()

    paths = list(args.qrels)
    if args.dir:
        paths += sorted(
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if f.endswith(".txt")
        )

    if not paths:
        parser.error("Provide at least one qrel file or use --dir.")

    ref_qrel = load_qrel(args.ref) if args.ref else None
    ref_name = os.path.basename(args.ref) if args.ref else ""

    all_stats = []
    for path in paths:
        qrel = load_qrel(path)
        name = os.path.basename(path)
        s = compute_stats(qrel, name=name)
        if ref_qrel is not None:
            add_comparison(s, qrel, ref_qrel, ref_name=ref_name)
        all_stats.append(s)

    if args.json:
        print(json.dumps([asdict(s) for s in all_stats], indent=2))
    else:
        print_stats_table(all_stats, show_comparison=ref_qrel is not None)


if __name__ == "__main__":
    main()
