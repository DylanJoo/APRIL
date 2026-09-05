import json
import sys
from collections import defaultdict
from pathlib import Path


def dedup_jsonl(filepath, dry_run=False):
    path = Path(filepath)
    lines = path.read_text().splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if line:
            records.append(json.loads(line))

    # Group by all fields except 'exp'
    key_fields = [k for k in records[0].keys() if k != "exp"]
    groups = defaultdict(list)
    for i, rec in enumerate(records):
        key = tuple(rec[f] for f in key_fields)
        groups[key].append((i, rec))

    to_remove = set()

    print("=== Exact duplicates (same exp, removing extras) ===")
    found_exact = False
    for key, entries in groups.items():
        if len(entries) < 2:
            continue
        seen_exps = {}
        for idx, rec in entries:
            exp_val = rec["exp"]
            if exp_val in seen_exps:
                found_exact = True
                print(f"  line {idx+1}: {json.dumps(rec)} <-- REMOVE (dup of line {seen_exps[exp_val]+1})")
                to_remove.add(idx)
            else:
                seen_exps[exp_val] = idx
    if not found_exact:
        print("  None found.")

    print("\n=== Redundant lines (null exp superseded by Llama exp) ===")
    found_null = False
    for key, entries in groups.items():
        if len(entries) < 2:
            continue
        active = [(idx, rec) for idx, rec in entries if idx not in to_remove]
        exps = {rec["exp"] for _, rec in active}
        has_llama = any(e is not None and "Llama" in str(e) for e in exps)
        has_null = None in exps
        if has_llama and has_null:
            found_null = True
            for idx, rec in active:
                marker = " <-- REMOVE (null exp)" if rec["exp"] is None else " <-- KEEP (Llama exp)"
                print(f"  line {idx+1}: {json.dumps(rec)}{marker}")
                if rec["exp"] is None:
                    to_remove.add(idx)
    if not found_null:
        print("  None found.")

    if not found_exact and not found_null:
        return

    print(f"\nTotal lines to remove: {len(to_remove)}")
    if dry_run:
        print("Dry run — no changes written.")
        return

    kept = [rec for i, rec in enumerate(records) if i not in to_remove]
    path.write_text("\n".join(json.dumps(r) for r in kept) + "\n")
    print(f"Written {len(kept)} lines to {filepath}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to .jsonl file")
    parser.add_argument("--dry-run", action="store_true", help="List only, no writes")
    args = parser.parse_args()
    dedup_jsonl(args.filepath, dry_run=args.dry_run)
