"""tune_bm25.py -- grid-search BM25 (k1, b, stemming) on the active dataset.

For the dataset named in config.yaml, tries every combination of the
`bm25_tuning` grid (k1 x b x use_stemming), retrieves all queries with each,
scores them by mean NDCG@eval_k (retrieval.eval_k, default 10 -- the project's
PRIMARY metric; independent of retrieval.top_k, which is the fusion candidate
pool used elsewhere and not needed here), and reports the best combination.

Outputs (to results/bm25_tuning/):
    <dataset>_bm25_tuning.csv  -- every combination and its mean NDCG@100
    <dataset>_bm25_best.json   -- the winning k1/b/use_stemming

Copy the winner into the `bm25` block of config.yaml manually.

Efficiency: tokenisation depends only on stemming, so the corpus is tokenised
once per stemming option; only the (k1, b) re-indexing runs inside the grid.
Only BM25 is needed here -- no embeddings, no GPU.
"""

import os
import sys
import gc
import json
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from tqdm import tqdm

import bm25s
try:
    import Stemmer
except ImportError:  # PyStemmer optional; stemming combos skipped with a warning
    Stemmer = None

from beir.datasets.data_loader import GenericDataLoader

from utils import load_config, get_paths, dataset_dir


# --- text + NDCG (kept local so this script needs no torch/matplotlib) ------
def build_doc_text(doc):
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return (title + " " + text).strip() if title else text


def ndcg_at_k(ranked_ids, rels, k):
    """NDCG@k with gain 2^rel - 1. None when the query has no relevant docs."""
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = sum((2.0 ** g - 1.0) / np.log2(i + 2.0) for i, g in enumerate(ideal))
    if idcg == 0.0:
        return None
    dcg = 0.0
    for i, d in enumerate(ranked_ids[:k]):
        g = rels.get(d, 0)
        if g > 0:
            dcg += (2.0 ** g - 1.0) / np.log2(i + 2.0)
    return dcg / idcg


def mean_ndcg(idx, qids, doc_ids, qrels, k):
    """Average NDCG@k over queries that have at least one relevant doc."""
    total, n = 0.0, 0
    for qi, q in enumerate(qids):
        rels = {d: int(g) for d, g in qrels[q].items() if int(g) > 0}
        if not rels:
            continue
        ranked = [doc_ids[int(j)] for j in idx[qi]]
        nd = ndcg_at_k(ranked, rels, k)
        if nd is None:
            continue
        total += nd
        n += 1
    return (total / n if n else 0.0), n


def main():
    config = load_config()
    paths = get_paths(config)
    name = config["dataset"]
    split = config.get("split", "test")
    method = config["bm25"].get("method", "lucene")
    k = config["retrieval"].get("eval_k", 10)   # PRIMARY metric: NDCG@eval_k (default 10)

    grid = config["bm25_tuning"]
    k1_vals = list(grid["k1"])
    b_vals = list(grid["b"])
    stem_vals = list(grid["use_stemming"])

    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_dir(paths, name)).load(split=split)
    doc_ids = list(corpus.keys())
    doc_texts = [build_doc_text(corpus[d]) for d in doc_ids]
    del corpus
    gc.collect()

    qids = [q for q in queries if q in qrels and len(qrels[q]) > 0]
    q_texts = [queries[q] for q in qids]
    k = min(k, len(doc_ids))

    n_combos = len(k1_vals) * len(b_vals) * len(stem_vals)
    print(f"[tune] '{name}': {len(doc_ids):,} docs | {len(qids):,} queries | "
          f"grid = {len(k1_vals)}xk1 * {len(b_vals)}xb * {len(stem_vals)}xstem "
          f"= {n_combos} runs | metric = mean NDCG@{k}")

    records = []
    pbar = tqdm(total=n_combos, desc="[tune] BM25 grid")
    for stem in stem_vals:
        stemmer = None
        if stem:
            if Stemmer is None:
                print("[tune] PyStemmer missing -- skipping use_stemming=true combos.")
                pbar.update(len(k1_vals) * len(b_vals))
                continue
            stemmer = Stemmer.Stemmer("english")

        # tokenise once per stemming option
        corpus_tokens = bm25s.tokenize(doc_texts, stopwords="en", stemmer=stemmer, show_progress=False)
        q_tokens = bm25s.tokenize(q_texts, stopwords="en", stemmer=stemmer, show_progress=False)

        for k1, b in itertools.product(k1_vals, b_vals):
            retriever = bm25s.BM25(method=method, k1=k1, b=b)
            retriever.index(corpus_tokens, show_progress=False)
            idx, _ = retriever.retrieve(q_tokens, k=k, show_progress=False)
            score, n_q = mean_ndcg(idx, qids, doc_ids, qrels, k)
            records.append({
                "dataset": name, "method": method,
                "k1": k1, "b": b, "use_stemming": bool(stem),
                f"mean_ndcg@{k}": round(score, 6), "n_queries": n_q,
            })
            pbar.set_postfix(k1=k1, b=b, stem=int(stem), ndcg=round(score, 4))
            pbar.update(1)

        del corpus_tokens, q_tokens
        gc.collect()
    pbar.close()

    if not records:
        print("[tune] no combinations evaluated (grid empty or all skipped).")
        return

    score_col = f"mean_ndcg@{k}"
    df = pd.DataFrame(records).sort_values(score_col, ascending=False).reset_index(drop=True)
    out_csv = os.path.join(paths["bm25_tuning"], f"{name}_bm25_tuning.csv")
    df.to_csv(out_csv, index=False)

    best = df.iloc[0].to_dict()
    out_json = os.path.join(paths["bm25_tuning"], f"{name}_bm25_best.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"\n[tune] wrote {out_csv}")
    print(f"[tune] wrote {out_json}")
    print(f"[tune] top 5 combinations:\n{df.head(5).to_string(index=False)}")
    print(f"\n[tune] BEST -> k1={best['k1']}, b={best['b']}, "
          f"use_stemming={best['use_stemming']}  ({score_col}={best[score_col]})")
    print("[tune] copy into config.yaml `bm25`:\n"
          f"  k1: {best['k1']}\n  b: {best['b']}\n  use_stemming: {str(best['use_stemming']).lower()}")


if __name__ == "__main__":
    main()
