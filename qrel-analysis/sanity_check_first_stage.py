"""
Sanity check: all reranking run files that share the same first-stage retriever
and dataset must have identical candidate document sets per query.

Filenames must follow the pattern:
    run.{benchmark}.{first_stage}-rerank-{reranker}.{dataset}.txt

The script groups files by (first_stage, dataset), then within each group
verifies that every run exposes the same {qid: set(docids)} mapping up to
--topk candidates.

Exit code: 0 if all groups pass, 1 if any group fails.

Output lines containing "OK ", "FAIL", "All first-stage", or "Could not detect"
are used by check_first_stage_consistency.sh via grep.
"""

import argparse
import re
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def parse_trec_run(filepath, topk=None):
    """Return {qid: list[docid]} keeping only top-k per query by rank field."""
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
        result[qid] = ranked[:topk] if topk else ranked
    return result


def parse_group_key(filepath):
    """
    Extract a group label from a run filepath.

    Filename must match:
        run.{benchmark}.{first_stage}-rerank-{reranker}.{dataset}.txt

    The parent directory is included when it looks like a sample directory
    (e.g. 'sample-1'), because different samples cover different query subsets
    and must be checked independently.

    Returns (label, first_stage) or (None, None) on parse failure.
    """
    path_parts = filepath.rsplit("/", 2)
    fname = path_parts[-1]
    parent = path_parts[-2] if len(path_parts) >= 2 else ""

    m = re.match(r'^run\.([^.]+)\.(.+?)-rerank-.+\.([^.]+)\.txt$', fname)
    if not m:
        return None, None

    benchmark, first_stage, dataset = m.group(1), m.group(2), m.group(3)
    label = f"{benchmark}/{first_stage}/{dataset}"
    if re.match(r'^sample-\d+$', parent):
        label = f"{label}/{parent}"
    return label, first_stage


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_group(name, fps, topk):
    """
    Check that all run files in a group have identical candidate sets.
    Returns True if consistent, False otherwise.
    """
    sep = "-" * 60
    print(sep)
    print(f"Group: {name}  ({len(fps)} run file(s))")

    runs = {}
    for fp in fps:
        # Use last two path components to disambiguate sample-N subdirs
        parts = fp.rsplit("/", 2)
        label = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        try:
            runs[label] = parse_trec_run(fp, topk=topk)
        except OSError as e:
            print(f"  FAIL could not read {label}: {e}")
            return False

    names = list(runs.keys())

    if len(names) < 2:
        print(f"  OK  only 1 run file — nothing to compare")
        return True

    ref_name = names[0]
    ref = runs[ref_name]
    ref_qids = set(ref.keys())

    ok = True

    # Check 1: query sets identical
    qid_ok = True
    for name_ in names[1:]:
        other_qids = set(runs[name_].keys())
        if other_qids != ref_qids:
            qid_ok = False
            ok = False
            extra = sorted(other_qids - ref_qids)[:5]
            missing = sorted(ref_qids - other_qids)[:5]
            print(f"  FAIL query set mismatch: {name_} vs {ref_name}")
            if extra:
                print(f"       extra queries  : {extra}")
            if missing:
                print(f"       missing queries: {missing}")
    if qid_ok:
        pass
        # print(f"  OK  query sets identical ({len(ref_qids)} queries)")

    # Check 2: candidate sets identical per query (order-insensitive)
    common_qids = ref_qids.copy()
    for name_ in names[1:]:
        common_qids &= set(runs[name_].keys())

    ref_sets = {qid: set(ref[qid]) for qid in common_qids}
    n_mismatches = 0
    examples = []

    for name_ in names[1:]:
        for qid in sorted(common_qids):
            other_set = set(runs[name_][qid])
            if other_set != ref_sets[qid]:
                n_mismatches += 1
                ok = False
                if len(examples) < 3:
                    examples.append((qid, name_,
                                     sorted(ref_sets[qid] - other_set)[:3],
                                     sorted(other_set - ref_sets[qid])[:3]))

    if n_mismatches == 0:
        pass
        # print(f"  OK  candidate sets identical across all {len(names)} runs "
        #       f"(checked {len(common_qids)} queries × {len(names) - 1} pairs)")
    else:
        pass
        # print(f"  FAIL {n_mismatches} (query, run) pair(s) with mismatched candidate sets")
        # for qid, run_name, missing_docs, extra_docs in examples:
        #     print(f"       query={qid}  run={run_name}")
        #     if missing_docs:
        #         print(f"         missing from run : {missing_docs}")
        #     if extra_docs:
        #         print(f"         extra in run     : {extra_docs}")

    # Check 3: candidate counts per query (sanity on topk truncation)
    size_issues = []
    for qid in sorted(common_qids):
        ref_n = len(ref[qid])
        for name_ in names[1:]:
            other_n = len(runs[name_][qid])
            if other_n != ref_n and len(size_issues) < 3:
                size_issues.append((qid, name_, ref_n, other_n))

    if size_issues:
        print(f"  WARN candidate count differs for some queries:")
        for qid, name_, ref_n, other_n in size_issues:
            print(f"       query={qid}: {ref_name}={ref_n}  {name_}={other_n}")
    else:
        print(f"  OK  candidate counts consistent")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check that reranking runs share identical first-stage candidates.")
    parser.add_argument("--run_files", nargs="+", required=True,
                        help="TREC-format reranking run files to check")
    parser.add_argument("--topk", type=int, default=100,
                        help="Candidate depth used during reranking (default: 100)")
    args = parser.parse_args()

    sep = "=" * 60

    # Group files by (first_stage, benchmark, dataset)
    groups = defaultdict(list)
    unknown = []
    for fp in args.run_files:
        label, _ = parse_group_key(fp)
        if label is None:
            unknown.append(fp)
        else:
            groups[label].append(fp)

    if unknown:
        print(f"Could not detect first-stage for {len(unknown)} file(s):")
        for fp in unknown:
            print(f"  {fp}")

    if not groups:
        print("No run files with a recognisable filename pattern found.")
        sys.exit(1)

    all_ok = True
    for label in sorted(groups):
        passed = check_group(label, groups[label], args.topk)
        if not passed:
            all_ok = False

    print(sep)
    if unknown:
        print(f"Could not detect first-stage name for {len(unknown)} file(s).")
    if all_ok:
        print("All first-stage candidate sets consistent.")
        sys.exit(0)
    else:
        print("FAIL — inconsistencies detected (see above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
