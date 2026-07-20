"""core.py -- primitives shared by pipeline.py (sections 0-3) and sections.py (4-9).

BEIR I/O, retrieval metrics, the fusion functions, the alpha->NDCG curve, and
bootstrap helpers. Kept in one place so both halves of the pipeline use exactly
the same definitions (and to avoid a circular import).
"""

import os
import json

# Thread caps must precede numpy: n_jobs=-1 (=32) oversubscribes and HANGS on a
# shared many-core box (profiled: 8 threads 2.0 s/fit, 32 stalls).
N_THREADS = int(os.environ.get("PIPE_THREADS", "8"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(N_THREADS))

import numpy as np

from utils import processed_dir, build_doc_text

RRF_K = 60


# =========================================================================== #
# BEIR I/O
# =========================================================================== #
def read_corpus_texts(folder, canonical_ids):
    by_id = {}
    with open(os.path.join(folder, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            by_id[d["_id"]] = build_doc_text(d)
    return [by_id.get(i, "") for i in canonical_ids]


def read_queries(folder):
    q = {}
    with open(os.path.join(folder, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            q[d["_id"]] = d.get("text", "")
    return q


def read_qrels(folder, split):
    p = os.path.join(folder, "qrels", f"{split}.tsv")
    if not os.path.exists(p):
        return None
    out = {}
    with open(p, encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                out.setdefault(parts[0], {})[parts[1]] = int(parts[2])
    return out


def load_retrieval(paths, name, split, top_k):
    p = os.path.join(processed_dir(paths, name, create=False),
                     f"retrieval_{split}_top{top_k}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run section 3 (retrieve) first.")
    z = np.load(p, allow_pickle=False)
    return ([str(q) for q in z["qids"]], z["bm_idx"], z["bm_val"],
            z["dn_idx"], z["dn_val"], z["q_emb"])


# =========================================================================== #
# Metrics (gain 2^rel - 1, trec_eval / BEIR convention)
# =========================================================================== #
def ndcg(ranked, rels, k):
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = sum((2.0 ** g - 1.0) / np.log2(i + 2.0) for i, g in enumerate(ideal))
    if idcg == 0.0:
        return None
    dcg = sum((2.0 ** rels[d] - 1.0) / np.log2(i + 2.0)
              for i, d in enumerate(ranked[:k]) if rels.get(d, 0) > 0)
    return dcg / idcg


def mrr(ranked, rels, k):
    for i, d in enumerate(ranked[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at(ranked, rels, k):
    n = sum(1 for g in rels.values() if g > 0)
    if n == 0:
        return None
    return sum(1 for d in ranked[:k] if rels.get(d, 0) > 0) / n


# =========================================================================== #
# Fusion. Inputs: doc-row arrays (descending) + their RAW scores.
# =========================================================================== #
def _minmax(rows, sc):
    sc = np.asarray(sc, dtype=np.float64)
    lo, hi = sc.min(), sc.max()
    z = (sc - lo) / (hi - lo) if hi > lo else np.zeros_like(sc)
    return {int(d): float(v) for d, v in zip(rows, z)}


def _zscore(rows, sc):
    sc = np.asarray(sc, dtype=np.float64)
    mu, sd = sc.mean(), sc.std()
    z = (sc - mu) / sd if sd > 0 else np.zeros_like(sc)
    z = z - z.min()                      # shift so "absent" (0) is the floor
    return {int(d): float(v) for d, v in zip(rows, z)}


NORMALIZERS = {"minmax": _minmax, "zscore": _zscore}


def fuse_score(bm_r, dn_r, bm_s, dn_s, alpha, N, norm="minmax"):
    """PRIMARY fusion: convex combination of per-query normalised scores.

    Unlike rank fusion this KEEPS score magnitude -- the information saying
    "doc A and B are both excellent, C is junk" rather than merely "A > B > C".
    """
    f = NORMALIZERS[norm or "minmax"]
    nb, nd = f(bm_r, bm_s), f(dn_r, dn_s)
    docs = list(set(nb) | set(nd))
    s = np.array([alpha * nb.get(d, 0.0) + (1.0 - alpha) * nd.get(d, 0.0) for d in docs])
    return [docs[i] for i in np.argsort(-s, kind="stable")]


def fuse_borda(bm_r, dn_r, bm_s, dn_s, alpha, N, norm=None):
    """BASELINE: linear rank points, N - rank0 (missing from a list -> 0)."""
    rb = {int(d): r for r, d in enumerate(bm_r)}
    rd = {int(d): r for r, d in enumerate(dn_r)}
    docs = list(set(rb) | set(rd))
    s = np.array([alpha * (N - rb[d] if d in rb else 0.0) +
                  (1.0 - alpha) * (N - rd[d] if d in rd else 0.0) for d in docs])
    return [docs[i] for i in np.argsort(-s, kind="stable")]


def fuse_rrf(bm_r, dn_r, bm_s, dn_s, alpha, N, norm=None):
    """BASELINE: weighted reciprocal rank fusion; alpha=0.5 is plain RRF."""
    rb = {int(d): r for r, d in enumerate(bm_r)}
    rd = {int(d): r for r, d in enumerate(dn_r)}
    docs = list(set(rb) | set(rd))
    s = np.array([alpha * (1.0 / (RRF_K + rb[d] + 1) if d in rb else 0.0) +
                  (1.0 - alpha) * (1.0 / (RRF_K + rd[d] + 1) if d in rd else 0.0)
                  for d in docs])
    return [docs[i] for i in np.argsort(-s, kind="stable")]


FUSERS = {"score": fuse_score, "borda": fuse_borda, "rrf": fuse_rrf}


def fusion_arrays(bm_r, dn_r, bm_s, dn_s, fusion, N, norm="minmax"):
    """Per-doc contribution of each retriever, aligned to the union of the lists.

    Neither side depends on alpha, so sweeping the alpha grid becomes a
    VECTORISED weighted sum instead of 101 full re-fusions (which re-normalised
    both 1000-doc lists every time -- 101x redundant work per query).
    """
    if fusion == "score":
        f = NORMALIZERS[norm or "minmax"]
        nb, nd = f(bm_r, bm_s), f(dn_r, dn_s)
    elif fusion == "borda":
        nb = {int(d): float(N - r) for r, d in enumerate(bm_r)}
        nd = {int(d): float(N - r) for r, d in enumerate(dn_r)}
    elif fusion == "rrf":
        nb = {int(d): 1.0 / (RRF_K + r + 1) for r, d in enumerate(bm_r)}
        nd = {int(d): 1.0 / (RRF_K + r + 1) for r, d in enumerate(dn_r)}
    else:
        raise ValueError(fusion)
    docs = list(set(nb) | set(nd))
    va = np.fromiter((nb.get(d, 0.0) for d in docs), dtype=np.float64, count=len(docs))
    vb = np.fromiter((nd.get(d, 0.0) for d in docs), dtype=np.float64, count=len(docs))
    return docs, va, vb


def topk_ids(docs, s, k):
    """Top-k doc ids by score. argpartition is O(n) vs argsort's O(n log n);
    only the k winners are then ordered."""
    if len(s) <= k:
        order = np.argsort(-s, kind="stable")
    else:
        part = np.argpartition(-s, k)[:k]
        order = part[np.argsort(-s[part], kind="stable")]
    return [docs[i] for i in order[:k]]


def alpha_curve(bm_r, dn_r, bm_s, dn_s, rels, cid, alphas, N, k, fusion, norm):
    """NDCG@k at EVERY alpha on the grid.

    Storing the whole curve (not just the argmax) is what lets any predicted
    alpha be scored later by table lookup instead of re-running retrieval, and
    it yields the per-query alpha sensitivity. np.argmax -> lowest alpha wins ties.
    """
    docs, va, vb = fusion_arrays(bm_r, dn_r, bm_s, dn_s, fusion, N, norm)
    curve = np.empty(len(alphas), dtype=np.float32)
    for i, a in enumerate(alphas):
        v = ndcg([cid[j] for j in topk_ids(docs, a * va + (1.0 - a) * vb, k)], rels, k)
        if v is None:
            return None, None, None
        curve[i] = v
    b = int(np.argmax(curve))
    return curve, float(alphas[b]), float(curve[b])


# =========================================================================== #
# Bootstrap
# =========================================================================== #
def bootstrap_ci(x, n, seed):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_bootstrap(a, b, n, seed):
    """Paired bootstrap of (a-b) over the SAME queries: far more powerful than
    comparing independent CIs, which can overlap while one method still wins
    query-by-query."""
    d = a - b
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
