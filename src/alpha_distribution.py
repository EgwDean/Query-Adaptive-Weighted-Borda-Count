"""alpha_distribution.py -- per-query oracle alpha for Weighted Borda Count.

For the active dataset (config.yaml):
  1. Retrieve top-k with BM25 (lexical) and the cached dense embeddings.
  2. For every query, fuse the two ranked lists with Weighted Borda Count
         score(d) = alpha*(N - rank_sparse(d)) + (1-alpha)*(N - rank_dense(d))
     over a grid of alpha in [0, 1], and record the alpha that maximises
     NDCG@k (lowest alpha wins ties) as that query's ORACLE ALPHA.
         alpha = 1 -> pure BM25 (lexical) ; alpha = 0 -> pure dense (semantic).
  3. Save per-query results, a per-dataset boxplot, and -- across every dataset
     processed so far -- a combined boxplot and a summary table ranking
     datasets by alpha spread / closeness to 0.5.

The dataset with the highest alpha spread (or median nearest 0.5) is the
strongest lexical+semantic test bed for the adaptive system.
"""

import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import bm25s
try:
    import Stemmer
except ImportError:  # PyStemmer optional; stemming silently disabled if absent
    Stemmer = None

from beir.datasets.data_loader import GenericDataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_config, get_paths, dataset_dir, processed_dir


# --------------------------------------------------------------------------- #
# Text + metrics
# --------------------------------------------------------------------------- #
def build_doc_text(doc):
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return (title + " " + text).strip() if title else text


def dcg_at_k(ranked_ids, rels, k):
    """Discounted cumulative gain with gain = 2^rel - 1 (trec_eval / BEIR)."""
    s = 0.0
    for i, d in enumerate(ranked_ids[:k]):
        g = rels.get(d, 0)
        if g > 0:
            s += (2.0 ** g - 1.0) / np.log2(i + 2.0)
    return s


def ndcg_at_k(ranked_ids, rels, k):
    """NDCG@k. Returns None when the query has no relevant docs (IDCG == 0)."""
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = sum((2.0 ** g - 1.0) / np.log2(i + 2.0) for i, g in enumerate(ideal))
    if idcg == 0.0:
        return None
    return dcg_at_k(ranked_ids, rels, k) / idcg


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def bm25_retrieve(corpus, doc_ids, queries, qids, conf, top_k):
    """Return {qid: [doc_id, ...]} top-k BM25 lists (rank 0 = best)."""
    texts = [build_doc_text(corpus[d]) for d in doc_ids]

    stemmer = None
    if conf.get("use_stemming", False):
        if Stemmer is not None:
            stemmer = Stemmer.Stemmer("english")
        else:
            print("[alpha] PyStemmer not installed -- BM25 stemming disabled.")

    print(f"[alpha] BM25: tokenising {len(texts):,} docs / building index "
          f"(k1={conf['k1']}, b={conf['b']}, method={conf.get('method', 'lucene')})")
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=True)
    retriever = bm25s.BM25(method=conf.get("method", "lucene"), k1=conf["k1"], b=conf["b"])
    retriever.index(corpus_tokens, show_progress=True)

    q_texts = [queries[q] for q in qids]
    q_tokens = bm25s.tokenize(q_texts, stopwords="en", stemmer=stemmer, show_progress=False)
    k = min(top_k, len(doc_ids))
    idx, _ = retriever.retrieve(q_tokens, k=k, show_progress=True)  # (nq, k) corpus indices

    out = {}
    for qi, q in enumerate(qids):
        out[q] = [doc_ids[int(j)] for j in idx[qi]]
    return out


def dense_retrieve(name, paths, qids, top_k, device):
    """Return {qid: [doc_id, ...]} top-k dense (cosine) lists from cached embeddings."""
    pdir = processed_dir(paths, name, create=False)
    corpus_emb = np.load(os.path.join(pdir, "corpus_emb.npy"))
    with open(os.path.join(pdir, "corpus_ids.json"), encoding="utf-8") as f:
        c_ids = json.load(f)
    q_emb_all = np.load(os.path.join(pdir, "query_emb.npy"))
    with open(os.path.join(pdir, "query_ids.json"), encoding="utf-8") as f:
        q_ids_all = json.load(f)
    q_pos = {q: i for i, q in enumerate(q_ids_all)}

    dev = "cpu"
    if device != "cpu" and torch.cuda.is_available():
        dev = "cuda"
    elif device != "cpu":
        print("[alpha] CUDA not available -- dense retrieval on CPU.")

    C = torch.from_numpy(corpus_emb).to(dev)                  # (Ndoc, dim), normalised
    Q = torch.from_numpy(q_emb_all[[q_pos[q] for q in qids]]).to(dev)
    k = min(top_k, len(c_ids))

    out = {}
    bs = 256
    for start in tqdm(range(0, len(qids), bs), desc="[alpha] dense retrieve"):
        qb = Q[start:start + bs]
        sims = qb @ C.T                                       # cosine (embeddings normalised)
        _, topi = torch.topk(sims, k=k, dim=1)
        topi = topi.cpu().numpy()
        for r, qi in enumerate(range(start, min(start + bs, len(qids)))):
            out[qids[qi]] = [c_ids[int(j)] for j in topi[r]]
    return out


# --------------------------------------------------------------------------- #
# Weighted Borda Count + oracle alpha
# --------------------------------------------------------------------------- #
def borda_points(rank_list, N):
    """Points per doc: N - rank0 (best -> N, last -> 1). Missing docs score 0."""
    return {d: N - r for r, d in enumerate(rank_list)}


def oracle_alpha(sparse_list, dense_list, rels, N, alphas, k):
    """Best alpha (lowest wins ties) and its NDCG@k, or (None, None) if no rels."""
    sp = borda_points(sparse_list, N)
    dn = borda_points(dense_list, N)
    docs = list(set(sp) | set(dn))
    sp_v = np.array([sp.get(d, 0) for d in docs], dtype=np.float64)
    dn_v = np.array([dn.get(d, 0) for d in docs], dtype=np.float64)

    best_a, best_ndcg = None, -1.0
    for a in alphas:
        scores = a * sp_v + (1.0 - a) * dn_v
        order = np.argsort(-scores, kind="stable")           # stable -> deterministic ties
        ranked = [docs[i] for i in order[:k]]
        nd = ndcg_at_k(ranked, rels, k)
        if nd is None:
            return None, None
        if nd > best_ndcg + 1e-12:                           # strict -> first (lowest) alpha wins
            best_ndcg, best_a = nd, float(a)
    return best_a, best_ndcg


# --------------------------------------------------------------------------- #
# Plots + summary
# --------------------------------------------------------------------------- #
def plot_single(df, name, paths):
    iqr = df["alpha"].quantile(0.75) - df["alpha"].quantile(0.25)
    plt.figure(figsize=(4, 5))
    sns.boxplot(y=df["alpha"], color="#69b3a2", width=0.4)
    sns.stripplot(y=df["alpha"], color="black", size=2, alpha=0.25, jitter=0.25)
    plt.axhline(0.5, ls="--", color="red", lw=1)
    plt.ylim(-0.03, 1.03)
    plt.ylabel("oracle alpha   (1 = BM25 / lexical,  0 = dense / semantic)")
    plt.title(f"{name}\nmedian={df['alpha'].median():.2f}  IQR={iqr:.2f}")
    plt.tight_layout()
    out = os.path.join(paths["results"], f"{name}_alpha_boxplot.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[alpha] wrote {out}")


def plot_combined(paths):
    files = sorted(glob.glob(os.path.join(paths["results"], "*_alpha.csv")))
    if not files:
        return
    alldf = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    order = alldf.groupby("dataset")["alpha"].median().sort_values().index.tolist()
    plt.figure(figsize=(max(6, 1.3 * len(order)), 6))
    sns.boxplot(data=alldf, x="dataset", y="alpha", order=order, color="#69b3a2")
    plt.axhline(0.5, ls="--", color="red", lw=1)
    plt.ylim(-0.03, 1.03)
    plt.ylabel("oracle alpha   (1 = BM25 / lexical,  0 = dense / semantic)")
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.title("Oracle alpha distribution per dataset")
    plt.tight_layout()
    out = os.path.join(paths["results"], "combined_alpha_boxplot.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[alpha] wrote {out}")


def write_summary(paths):
    files = sorted(glob.glob(os.path.join(paths["results"], "*_alpha.csv")))
    if not files:
        return
    recs = []
    for f in files:
        d = pd.read_csv(f)
        a = d["alpha"]
        q1, q3 = a.quantile(0.25), a.quantile(0.75)
        recs.append({
            "dataset": d["dataset"].iloc[0],
            "n_queries": len(d),
            "alpha_mean": round(a.mean(), 4),
            "alpha_median": round(a.median(), 4),
            "alpha_std": round(a.std(), 4),
            "alpha_iqr": round(q3 - q1, 4),
            "dist_from_0.5": round(abs(a.median() - 0.5), 4),
            "bm25_ndcg": round(d["bm25_ndcg"].mean(), 4),
            "dense_ndcg": round(d["dense_ndcg"].mean(), 4),
            "oracle_ndcg": round(d["oracle_ndcg"].mean(), 4),
        })
    s = pd.DataFrame(recs).sort_values("alpha_iqr", ascending=False).reset_index(drop=True)
    out = os.path.join(paths["results"], "alpha_summary.csv")
    s.to_csv(out, index=False)
    print(f"[alpha] wrote {out}\n")
    print(s.to_string(index=False))
    most_spread = s.iloc[0]["dataset"]
    most_balanced = s.loc[s["dist_from_0.5"].idxmin(), "dataset"]
    print(f"\n[alpha] highest alpha spread (IQR): {most_spread}")
    print(f"[alpha] median closest to 0.5     : {most_balanced}")


# --------------------------------------------------------------------------- #
def main():
    config = load_config()
    paths = get_paths(config)
    name = config["dataset"]
    split = config.get("split", "test")
    N = config["borda"]["N"]
    k = config["retrieval"]["top_k"]
    alphas = np.round(
        np.arange(config["borda"]["alpha_min"],
                  config["borda"]["alpha_max"] + 1e-9,
                  config["borda"]["alpha_step"]),
        4,
    )

    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_dir(paths, name)).load(split=split)
    doc_ids = list(corpus.keys())
    qids = [q for q in queries if q in qrels and len(qrels[q]) > 0]
    print(f"[alpha] '{name}': {len(doc_ids):,} docs | {len(qids):,} queries with qrels "
          f"| alpha grid={len(alphas)} pts | top_k={k}")

    bm = bm25_retrieve(corpus, doc_ids, queries, qids, config["bm25"], k)
    dn = dense_retrieve(name, paths, qids, k, config["dense"].get("device", "cuda"))

    rows = []
    for q in tqdm(qids, desc="[alpha] fuse + score"):
        rels = {d: int(g) for d, g in qrels[q].items() if int(g) > 0}
        if not rels:
            continue
        a_star, nd = oracle_alpha(bm[q], dn[q], rels, N, alphas, k)
        if a_star is None:
            continue
        rows.append({
            "dataset": name,
            "qid": q,
            "alpha": a_star,
            "oracle_ndcg": nd,
            "bm25_ndcg": ndcg_at_k(bm[q], rels, k) or 0.0,
            "dense_ndcg": ndcg_at_k(dn[q], rels, k) or 0.0,
            "n_rel": len(rels),
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(paths["results"], f"{name}_alpha.csv")
    df.to_csv(out_csv, index=False)
    print(f"[alpha] wrote {out_csv}  ({len(df):,} queries)")

    plot_single(df, name, paths)
    plot_combined(paths)
    write_summary(paths)


if __name__ == "__main__":
    main()
