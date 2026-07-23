"""Report the split sizes of candidate BEIR datasets before committing to embedding.

    python src/probe_datasets.py msmarco nq climate-fever

Runs section 0 (download) only, then counts the corpus and the number of queries
with at least one positive judgement per split. Embedding a multi-million-doc
corpus costs hours, so check here first that a dataset actually has a usable
fit split and a test split large enough for significance.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, get_paths, dataset_dir
from core import read_qrels
import pipeline as P

MIN_FIT = 1000        # calibration needs >= MIN_QUERIES_PER_BIN * n_bins
MIN_TEST = 2000       # below this, small effects will not clear significance


def n_pos(folder, split):
    qr = read_qrels(folder, split)
    if qr is None:
        return None
    return sum(1 for q, d in qr.items() if any(int(v) > 0 for v in d.values()))


def corpus_size(folder):
    p = os.path.join(folder, "corpus.jsonl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--no-download", action="store_true",
                    help="only inspect datasets already on disk")
    args = ap.parse_args()

    cfg = load_config()
    paths = get_paths(cfg)
    print(f"{'dataset':18s} {'corpus':>11s} {'train':>8s} {'dev':>8s} {'test':>8s}   verdict")
    print("-" * 86)
    for ds in args.datasets:
        folder = dataset_dir(paths, ds)
        if not os.path.exists(os.path.join(folder, "corpus.jsonl")):
            if args.no_download:
                print(f"{ds:18s} {'(absent)':>11s}")
                continue
            try:
                c = dict(cfg, dataset=ds)
                print(f"[probe] downloading {ds} ...")
                P.sec_download(c, paths)
            except Exception as e:
                print(f"{ds:18s} download FAILED: {type(e).__name__}: {e}")
                continue
        n = {s: n_pos(folder, s) for s in ("train", "dev", "test")}
        cs = corpus_size(folder)

        def fmt(v):
            return f"{v:,}" if v else "-"

        # a usable cell needs a fit split and an adequately powered eval split
        fit = max([v for v in (n["train"], n["dev"]) if v] or [0])
        evalq = n["test"] or 0
        best_eval = max(evalq, n["dev"] or 0)
        if fit < MIN_FIT:
            v = "UNUSABLE: no split big enough to fit calibration"
        elif evalq >= MIN_TEST:
            v = "GOOD: fit + well-powered test"
        elif best_eval >= MIN_TEST:
            v = f"USABLE only if evaluated on dev ({n['dev']:,}); test is tiny"
        else:
            v = "WEAK: eval split too small for significance"
        print(f"{ds:18s} {fmt(cs):>11s} {fmt(n['train']):>8s} {fmt(n['dev']):>8s} "
              f"{fmt(n['test']):>8s}   {v}")

    print(f"\nthresholds: fit >= {MIN_FIT:,} queries, eval >= {MIN_TEST:,} queries")


if __name__ == "__main__":
    main()
