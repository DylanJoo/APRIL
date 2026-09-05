"""
qrel_correlation.py

Label-level correlation (Cohen's kappa) between LLM-derived auto-qrels
(rank@K binary strategy: top-K ranked docs = 1, rest = 0) and human qrels.

Sample alignment: the autoqrel's pool is the unit of analysis. For every
(qid, docid) the autoqrel judged, we look up the human qrel's label for
that same pair; if the human qrel never judged that doc, its label
defaults to 0 (non-relevant). The autoqrel's labels are always binary
(0/1) by construction; the human qrel keeps whatever grades it already
has (e.g. 0/1/2), so kappa is computed over the union of both label sets
rather than assuming a shared binary scale.

Usage
-----
# Default sweep: rank@10, all judge methods, top10 pool,
# across the 5 BEIR datasets shared with autoqrels-rank/.
python qrel_correlation.py

# Narrow it down
python qrel_correlation.py --datasets nfcorpus scidocs --methods judge umbrela

# A single explicit pair of files
python qrel_correlation.py --auto path/to/autollmqrel.rank@10.txt --human ~/runs-and-qrels/qrels/beir/qrels.beir.nfcorpus.txt

# JSON output
python qrel_correlation.py --json
"""

import argparse
import json
import os
from collections import Counter, defaultdict

DEFAULT_HUMAN_QREL_DIR = os.path.expanduser("~/runs-and-qrels/qrels/beir")
DEFAULT_AUTOQREL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autoqrels")

# trec-dl-2019/2020 human qrels aren't under the beir qrels dir (they're TREC
# DL judgments over the MS MARCO passage collection, not a BEIR dataset), so
# they need an explicit path template instead of qrels.beir.{dataset}.txt.
HUMAN_QREL_OVERRIDES = {
    "trec-dl-2019": os.path.expanduser("~/runs-and-qrels/qrels/msmarco-passage/qrels.msmarco-passage.trec-dl-2019.txt"),
    "trec-dl-2020": os.path.expanduser("~/runs-and-qrels/qrels/msmarco-passage/qrels.msmarco-passage.trec-dl-2020.txt"),
}

DEFAULT_DATASETS = ["dbpedia-entity", "nfcorpus", "scidocs", "trec-covid", "webis-touche2020",
                     "trec-dl-2019", "trec-dl-2020"]
DEFAULT_METHODS = ["judge", "judge_expr", "point", "rankgpt", "setmaxheaptopk", "umbrela"]
DEFAULT_POOLS = ["pool-40-systems-top10"]
RANK_CUTOFF = 10


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
# Cohen's kappa (general categories, aligned to the autoqrel's pool)
# ---------------------------------------------------------------------------

def align_to_auto_pool(auto_qrel, human_qrel):
    """For every (qid, docid) the autoqrel judged, pair it with the human
    qrel's label (0 if the human qrel never judged that doc)."""
    pairs = []
    matched = 0
    for qid, docs in auto_qrel.items():
        hdocs = human_qrel.get(qid, {})
        for did, auto_label in docs.items():
            human_label = hdocs.get(did, 0)
            if did in hdocs:
                matched += 1
            pairs.append((auto_label, human_label))
    return pairs, matched


def cohens_kappa(pairs):
    """Cohen's kappa over a list of (auto_label, human_label) pairs.
    Categories don't need to match between sides (e.g. auto in {0,1},
    human in {0,1,2}); the contingency table is built over the union."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "kappa": None, "po": None, "pe": None}

    counts = Counter(pairs)
    row_marg = Counter()  # auto label marginals
    col_marg = Counter()  # human label marginals
    agree = 0
    for (a, h), c in counts.items():
        row_marg[a] += c
        col_marg[h] += c
        if a == h:
            agree += c

    categories = set(row_marg) | set(col_marg)
    po = agree / n
    pe = sum((row_marg.get(k, 0) / n) * (col_marg.get(k, 0) / n) for k in categories)
    kappa = (po - pe) / (1 - pe) if pe != 1 else None

    return {
        "n": n,
        "kappa": round(kappa, 4) if kappa is not None else None,
        "po": round(po, 4),
        "pe": round(pe, 4),
    }


# ---------------------------------------------------------------------------
# Discovery + sweep
# ---------------------------------------------------------------------------

def find_auto_qrel(root, pool, method, dataset, rank_cutoff):
    path = os.path.join(root, f"{pool}-rerank-{method}", dataset, f"autollmqrel.rank@{rank_cutoff}.txt")
    return path if os.path.isfile(path) else None


def find_human_qrel(human_dir, dataset):
    return HUMAN_QREL_OVERRIDES.get(dataset, os.path.join(human_dir, f"qrels.beir.{dataset}.txt"))


def run_sweep(args):
    human_cache = {}
    rows = []

    for dataset in args.datasets:
        human_path = find_human_qrel(args.human_dir, dataset)
        if dataset not in human_cache:
            if not os.path.isfile(human_path):
                print(f"  [skip] human qrel not found: {human_path}")
                continue
            human_cache[dataset] = load_qrel(human_path)
        human_qrel = human_cache[dataset]

        for pool in args.pools:
            for method in args.methods:
                auto_path = find_auto_qrel(args.auto_root, pool, method, dataset, args.rank_cutoff)
                if auto_path is None:
                    continue
                auto_qrel = load_qrel(auto_path)  # already binary 0/1
                pairs, matched = align_to_auto_pool(auto_qrel, human_qrel)
                stats = cohens_kappa(pairs)
                rows.append({
                    "dataset": dataset,
                    "pool": pool,
                    "method": method,
                    "matched": matched,
                    **stats,
                })
    return rows


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def print_table(rows):
    """Pivoted view: one row per method, one column per dataset (kappa),
    plus a trailing macro-average column. Assumes a single pool; if more
    than one pool is present, dataset columns are collapsed across pools
    (a (method, dataset) cell keeps the last row seen for that pair)."""
    if not rows:
        print("No (dataset, pool, method) combinations found.")
        return

    methods, datasets = [], []
    kappa_map = {}
    for r in rows:
        m, d = r["method"], r["dataset"]
        if m not in methods:
            methods.append(m)
        if d not in datasets:
            datasets.append(d)
        kappa_map[(m, d)] = r["kappa"]

    def row_avg(m):
        vals = [kappa_map[(m, d)] for d in datasets if kappa_map.get((m, d)) is not None]
        return sum(vals) / len(vals) if vals else None

    methods = sorted(methods, key=lambda m: (row_avg(m) is None, -(row_avg(m) or 0)))

    method_w = max(18, max((len(m) for m in methods), default=0) + 2)
    col_w = max(12, max((len(d) for d in datasets), default=0) + 2)
    header = "  " + "{:<{}}".format("method", method_w) + \
        "".join("{:>{}}".format(d, col_w) for d in datasets) + \
        "{:>{}}".format("avg", col_w)
    print(header)
    print("-" * len(header))
    for m in methods:
        cells = [_fmt(kappa_map.get((m, d))) for d in datasets]
        avg_cell = _fmt(row_avg(m))
        print("  " + "{:<{}}".format(m, method_w) +
              "".join("{:>{}}".format(c, col_w) for c in cells) +
              "{:>{}}".format(avg_cell, col_w))


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_rows(rows):
    """Flat view: one line per (dataset, pool, method) row, followed by a
    macro-average-per-method summary. This is the format archived in
    kappa_rank{K}.beir.txt result files."""
    if not rows:
        print("No (dataset, pool, method) combinations found.")
        return

    header = "  " + "{:<24}{:<24}{:<24}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
        "dataset", "pool", "method", "n", "matched", "kappa", "po", "pe")
    print(header)
    for r in rows:
        print("  " + "{:<24}{:<24}{:<24}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
            r["dataset"], r["pool"], r["method"],
            r["n"], r["matched"], _fmt(r["kappa"]), _fmt(r["po"]), _fmt(r["pe"])))

    print("-" * len(header))
    print("  Macro-average kappa per method:")

    by_method = defaultdict(list)
    for r in rows:
        if r["kappa"] is not None:
            by_method[r["method"]].append(r["kappa"])
    avgs = {m: sum(vs) / len(vs) for m, vs in by_method.items()}
    for m in sorted(avgs, key=lambda m: -avgs[m]):
        print("    {:<24}{:.4f}  (n_combos={})".format(m, avgs[m], len(by_method[m])))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cohen's kappa between rank@K auto-qrels and human BEIR qrels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--auto", type=str, default=None, help="Single auto-qrel file (bypasses sweep).")
    parser.add_argument("--human", type=str, default=None, help="Single human qrel file (used with --auto).")

    parser.add_argument("--auto-root", type=str, default=DEFAULT_AUTOQREL_ROOT,
                         help="Root dir containing pool-40-systems-top{K}-rerank-{method}/{dataset}/ subdirs.")
    parser.add_argument("--human-dir", type=str, default=DEFAULT_HUMAN_QREL_DIR,
                         help="Dir containing qrels.beir.{dataset}.txt files.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--pools", nargs="+", default=DEFAULT_POOLS)
    parser.add_argument("--rank-cutoff", type=int, default=RANK_CUTOFF,
                         help="Which rank@K auto-qrel to compare (default: 10).")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--rows", action="store_true",
                         help="Flat per-row report (the format archived in kappa_rank{K}.beir.txt).")
    args = parser.parse_args()

    if args.auto:
        if not args.human:
            parser.error("--auto requires --human")
        auto_qrel = load_qrel(args.auto)
        human_qrel = load_qrel(args.human)
        pairs, matched = align_to_auto_pool(auto_qrel, human_qrel)
        stats = cohens_kappa(pairs)
        row = {"dataset": os.path.basename(args.human), "pool": "-", "method": os.path.basename(args.auto),
               "matched": matched, **stats}
        rows = [row]
    else:
        rows = run_sweep(args)

    if args.json:
        print(json.dumps(rows, indent=2))
    elif args.rows:
        print_rows(rows)
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
