"""Aggregate autollmrerank `*.stats.json` cost-report files into a summary table.

Each `python -m autollmrerank.wrapper` run writes a `<output_run>.stats.json`
file (see wrapper.py) containing call counts, prompt/completion word counts,
and timing, produced as a side effect of the LLM provider's `generate()`
calls during inference. This script just collects and reports those numbers.

Usage:
    python aggregate_cost_report.py --dir runs/<model>/cost_report [--benchmark beir] [--subset test] [--out summary.csv]
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics

FILENAME_RE = re.compile(
    r"run\.(?P<benchmark>[^.]+)\.(?P<retriever>.+)-rerank-(?P<method>[^.]+)\.(?P<subset>[^.]+)\.txt\.stats\.json$"
)


def parse_filename(path):
    m = FILENAME_RE.search(os.path.basename(path))
    if not m:
        return {}
    fields = m.groupdict()
    fields['pool'] = fields['retriever']
    return fields


def load_rows(directory, benchmark=None, subset=None):
    rows = []
    for path in glob.glob(os.path.join(directory, '**', '*.stats.json'), recursive=True):
        with open(path) as f:
            stats = json.load(f)
        meta = parse_filename(path)
        if benchmark and meta.get('benchmark') != benchmark:
            continue
        if subset and meta.get('subset') != subset:
            continue
        rows.append({**meta, **stats, 'path': path})
    return rows


def fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def print_table(rows):
    if not rows:
        print("No cost-report stats found.")
        return
    cols = ['benchmark', 'subset', 'pool', 'method', 'num_queries', 'num_prompts',
            'num_calls', 'time_sec', 'time_sec_per_query', 'prompt_words',
            'prompt_words_per_query', 'completion_words']
    widths = {c: max(len(c), *(len(fmt(r.get(c, ''))) for r in rows)) for c in cols}
    header = '  '.join(c.ljust(widths[c]) for c in cols)
    print(header)
    print('-' * len(header))
    for r in sorted(rows, key=lambda r: (r.get('benchmark', ''), r.get('pool', ''), r.get('method', ''))):
        print('  '.join(fmt(r.get(c, '')).ljust(widths[c]) for c in cols))


def print_pool_comparison(rows):
    by_method = {}
    for r in rows:
        by_method.setdefault(r.get('method'), {}).setdefault(r.get('pool'), []).append(r)

    print("\n=== Cost by method (avg across datasets) ===")
    header = f"{'method':<12} {'pool':<12} {'avg_time_sec/query':>20} {'avg_prompt_words/query':>24} {'n':>4}"
    print(header)
    print('-' * len(header))
    for method, pools in sorted(by_method.items()):
        for pool, rs in sorted(pools.items()):
            times = [r['time_sec_per_query'] for r in rs if 'time_sec_per_query' in r]
            words = [r['prompt_words_per_query'] for r in rs if 'prompt_words_per_query' in r]
            if not times or not words:
                continue
            print(f"{method:<12} {pool:<12} {statistics.mean(times):>20.3f} {statistics.mean(words):>24.1f} {len(rs):>4}")


def write_csv(rows, out_path):
    if not rows:
        return
    all_keys = sorted({k for r in rows for k in r.keys()})
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help="Directory to search recursively for *.stats.json")
    ap.add_argument('--benchmark', default=None, help="Filter to a single benchmark, e.g. beir")
    ap.add_argument('--subset', default=None, help="Filter to a single subset, e.g. test")
    ap.add_argument('--out', default=None, help="CSV output path (default: <dir>/cost_report_summary.csv)")
    args = ap.parse_args()

    rows = load_rows(args.dir, args.benchmark, args.subset)
    print_table(rows)
    print_pool_comparison(rows)
    write_csv(rows, args.out or os.path.join(args.dir, 'cost_report_summary.csv'))


if __name__ == '__main__':
    main()
