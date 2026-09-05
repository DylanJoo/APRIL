"""
Sanity checks for diverse_pooling.py output.

Checks:
  1. Query coverage        – every query has at least one pooled doc
  2. Pool size             – per-query unique doc counts (min/mean/max)
  3. Per-system unique contribution – docs each system adds exclusively
  4. Pairwise Jaccard overlap       – how redundant any two systems are
  5. Human-qrel recall     – fraction of known-relevant docs in the pool
                             (only when --qrel is provided)
"""

import argparse
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def parse_trec_run(filepath, topk=None):
    """Return {qid: set(docids)} keeping only top-k per query (by rank field)."""
    raw = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, docid, rank = parts[0], parts[2], int(parts[3])
            raw[qid].append((docid, rank))
    result = {}
    for qid, docs in raw.items():
        ranked = [d for d, _ in sorted(docs, key=lambda x: x[1])]
        result[qid] = set(ranked[:topk] if topk else ranked)
    return result


def parse_qrel(filepath, min_relevance=1):
    """Return {qid: set(relevant_docids)}."""
    relevant = defaultdict(set)
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, docid, rel = parts[0], parts[2], int(parts[3])
            if rel >= min_relevance:
                relevant[qid].add(docid)
    return dict(relevant)


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def check_query_coverage(pool, all_qids):
    missing = all_qids - set(pool.keys())
    empty = {q for q, docs in pool.items() if len(docs) == 0}
    return missing | empty


def pool_size_stats(pool):
    sizes = [len(docs) for docs in pool.values()]
    if not sizes:
        return {}
    return {
        "n_queries": len(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "mean": sum(sizes) / len(sizes),
        "median": sorted(sizes)[len(sizes) // 2],
    }


def per_system_unique(system_sets):
    """
    For each system, count docs that appear in that system's top-K
    but in no other system's top-K, averaged across queries.
    Returns {system_name: {"total_unique": int, "queries_with_unique": int}}
    """
    names = list(system_sets.keys())
    stats = {n: {"total_unique": 0, "queries_with_unique": 0} for n in names}
    all_qids = set()
    for docs_by_qid in system_sets.values():
        all_qids |= set(docs_by_qid.keys())

    for qid in all_qids:
        per_sys = {n: system_sets[n].get(qid, set()) for n in names}
        for name in names:
            others = set()
            for other_name, docs in per_sys.items():
                if other_name != name:
                    others |= docs
            unique = per_sys[name] - others
            stats[name]["total_unique"] += len(unique)
            if unique:
                stats[name]["queries_with_unique"] += 1
    return stats


def pairwise_jaccard(system_sets):
    """Return {(nameA, nameB): mean_jaccard} for all pairs."""
    names = list(system_sets.keys())
    all_qids = set()
    for docs_by_qid in system_sets.values():
        all_qids |= set(docs_by_qid.keys())

    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            jaccards = []
            for qid in all_qids:
                sa = system_sets[a].get(qid, set())
                sb = system_sets[b].get(qid, set())
                union = sa | sb
                if union:
                    jaccards.append(len(sa & sb) / len(union))
            results[(a, b)] = sum(jaccards) / len(jaccards) if jaccards else 0.0
    return results


def qrel_recall(pool, qrel):
    """
    Per query: recall = |pool ∩ relevant| / |relevant|.
    Returns (mean_recall, n_zero_recall_queries, n_evaluated_queries).
    """
    recalls = []
    zero_count = 0
    for qid, rel_docs in qrel.items():
        if not rel_docs:
            continue
        pooled = pool.get(qid, set())
        r = len(pooled & rel_docs) / len(rel_docs)
        recalls.append(r)
        if r == 0:
            zero_count += 1
    mean_r = sum(recalls) / len(recalls) if recalls else 0.0
    return mean_r, zero_count, len(recalls)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sanity-check a diverse-pooling run.")
    parser.add_argument("--run_files", nargs="+", required=True,
                        help="Individual system TREC run files (same as passed to diverse_pooling.py)")
    parser.add_argument("--pool", required=True,
                        help="Fused pool TREC run file (output of diverse_pooling.py)")
    parser.add_argument("--qrel", default=None,
                        help="Human qrel file (TREC format) for recall check")
    parser.add_argument("--topk", type=int, default=10,
                        help="Top-K per system used during pooling (default: 10)")
    parser.add_argument("--min_relevance", type=int, default=1,
                        help="Minimum relevance grade counted as relevant in qrel")
    args = parser.parse_args()

    sep = "-" * 60

    # Load data
    print(f"Loading pool from: {args.pool}")
    pool = parse_trec_run(args.pool)
    pool_docsets = {qid: set(docs) for qid, docs in pool.items()}

    print(f"Loading {len(args.run_files)} system run(s) ...")
    system_sets = {}
    for fp in args.run_files:
        name = fp.split("/")[-1]  # use filename as label
        system_sets[name] = parse_trec_run(fp, topk=args.topk)
        print(f"  {name}: {len(system_sets[name])} queries")

    all_qids = set(pool_docsets.keys())
    for docs_by_qid in system_sets.values():
        all_qids |= set(docs_by_qid.keys())

    print()

    # ------------------------------------------------------------------
    # Check 1: Query coverage
    # ------------------------------------------------------------------
    print(sep)
    print("CHECK 1: Query coverage")
    uncovered = check_query_coverage(pool_docsets, all_qids)
    if uncovered:
        print(f"  FAIL — {len(uncovered)} queries have no pool docs: {sorted(uncovered)[:10]}")
    else:
        print(f"  OK — all {len(all_qids)} queries have at least one pooled doc")

    # ------------------------------------------------------------------
    # Check 2: Pool size distribution
    # ------------------------------------------------------------------
    print(sep)
    print("CHECK 2: Pool size per query (unique docs)")
    stats = pool_size_stats(pool_docsets)
    print(f"  Queries : {stats['n_queries']}")
    print(f"  Min     : {stats['min']}")
    print(f"  Median  : {stats['median']}")
    print(f"  Mean    : {stats['mean']:.1f}")
    print(f"  Max     : {stats['max']}")
    max_possible = args.topk * len(args.run_files)
    mean_util = stats["mean"] / max_possible * 100
    print(f"  Utilisation vs. max possible ({max_possible}): {mean_util:.1f}%")

    # ------------------------------------------------------------------
    # Check 3: Per-system unique contribution
    # ------------------------------------------------------------------
    print(sep)
    print("CHECK 3: Per-system unique contribution (docs not in any other system's top-K)")
    unique_stats = per_system_unique(system_sets)
    n_q = len(all_qids)
    for name, s in unique_stats.items():
        avg_unique = s["total_unique"] / n_q if n_q else 0
        q_pct = s["queries_with_unique"] / n_q * 100 if n_q else 0
        flag = "  " if avg_unique > 0 else "!"
        print(f"  {flag} {name}")
        print(f"      avg unique docs/query : {avg_unique:.2f}")
        print(f"      queries with ≥1 unique: {s['queries_with_unique']}/{n_q} ({q_pct:.0f}%)")

    # ------------------------------------------------------------------
    # Check 4: Pairwise Jaccard overlap
    # ------------------------------------------------------------------
    print(sep)
    print("CHECK 4: Mean pairwise Jaccard overlap (lower = more diverse)")
    if len(system_sets) < 2:
        print("  Only one system — skip.")
    else:
        jaccards = pairwise_jaccard(system_sets)
        for (a, b), j in sorted(jaccards.items(), key=lambda x: -x[1]):
            flag = "!" if j > 0.5 else "  "
            print(f"  {flag} {a}  ↔  {b}  :  {j:.3f}")

    # ------------------------------------------------------------------
    # Check 5: Human-qrel recall (optional)
    # ------------------------------------------------------------------
    if args.qrel:
        print(sep)
        print("CHECK 5: Human-qrel recall of pool")
        qrel = parse_qrel(args.qrel, min_relevance=args.min_relevance)
        mean_r, zero_q, n_eval = qrel_recall(pool_docsets, qrel)
        print(f"  Queries evaluated : {n_eval}")
        print(f"  Mean recall       : {mean_r:.4f}")
        if zero_q > 0:
            print(f"  ! Queries with 0 recall: {zero_q} ({zero_q/n_eval*100:.1f}%)")
        else:
            print(f"  OK — no queries with zero recall")
    else:
        print(sep)
        print("CHECK 5: Human-qrel recall — skipped (pass --qrel to enable)")

    print(sep)
    print("Done.")


if __name__ == "__main__":
    main()
