"""pipeline.py -- run the full Phase-1 pipeline for the active dataset.

Runs, in order, the three stages for the dataset named in config.yaml:
    1. download.py            -> fetch the BEIR dataset
    2. embed.py               -> embed docs + queries (all-mpnet-base-v2)
    3. alpha_distribution.py  -> BM25 + dense + oracle alpha + boxplots

This is just a convenience wrapper; each stage can still be run on its own.
Change `dataset:` in config.yaml, then:  python src/pipeline.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import download
import embed
import alpha_distribution

from utils import load_config

STAGES = [
    ("download", download.main),
    ("embed", embed.main),
    ("alpha_distribution", alpha_distribution.main),
]


def main():
    name = load_config().get("dataset", "?")
    print(f"\n########## PIPELINE START: dataset='{name}' ##########")
    t_all = time.perf_counter()

    for i, (label, fn) in enumerate(STAGES, start=1):
        print(f"\n---------- [{i}/{len(STAGES)}] {label} ----------")
        t0 = time.perf_counter()
        fn()
        print(f"---------- {label} done in {time.perf_counter() - t0:.1f}s ----------")

    print(f"\n########## PIPELINE DONE: dataset='{name}' "
          f"({time.perf_counter() - t_all:.1f}s total) ##########")


if __name__ == "__main__":
    main()
