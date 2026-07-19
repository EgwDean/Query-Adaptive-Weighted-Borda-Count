"""benchmark.py -- Stage 5a: the core benchmark. Opens the TEST split ONCE.

Evaluates every Tier-0/Tier-1 method from docs/comparison_methods.md against the
frozen router from final_fit.py, on the held-out test split.

Methods
-------
  1  BM25                      (tuned k1/b)                       Tier 0 floor
  2  Dense                     (all-mpnet-base-v2)                Tier 0 floor
  3  Static Borda alpha=0.5    naive equal fusion                 Tier 1
  4  Static Borda alpha*       ONE global alpha                   Tier 1  <- PRIMARY BASELINE
  5  RRF k=60                  reciprocal rank fusion, untuned    Tier 1
  6  Weighted RRF alpha*       ONE global alpha                   Tier 1
  7  Norm-score fusion alpha*  min-max score convex combination   Tier 1
  8  ROUTER (ours)             per-query alpha -> Weighted Borda  Tier 2
  9  Oracle alpha              per-query best alpha               ceiling

Why this script re-runs retrieval
---------------------------------
The cached alpha->NDCG curve only covers Weighted Borda, so RRF and score fusion
(different fusion FUNCTIONS) cannot be read off it -- they need the actual ranked
lists and raw scores. We therefore re-retrieve BM25 + dense for dev and test.

Tuning discipline
-----------------
All global-alpha baselines (#4, #6, #7) are tuned on **dev** and evaluated on
**test**, so every baseline gets exactly the same tuning opportunity the router
had (fit on train, selected on dev). Test is never used to choose anything.

The comparison that matters: #8 vs #4. Beating alpha=0.5 is not a result;
beating the best single global alpha is. Also reported: #5/#6 vs #3/#4 tests the
project's core design claim -- that Borda's LINEAR rank scoring gives alpha more
leverage than RRF's saturating 1/(k+rank).

Outputs (results/router_final/):
    <ds>_benchmark.csv           -- one row per method x metric, with CIs
    <ds>_benchmark_per_query.csv -- per-query NDCG@eval_k for every method
    <ds>_benchmark.json          -- headline numbers + significance vs the baseline
"""

import os
import sys
import gc
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
import torch
import bm25s
try:
    import Stemmer
except ImportError:
    Stemmer = None

from utils import load_config, get_paths, dataset_dir, processed_dir
from screen_routers import predict_alpha, apply_calibration, paired_bootstrap, bootstrap_ci
from create_dataset import (read_corpus_texts, read_queries, read_qrels,
                            dense_retrieve_scores, _resolve_device)

RRF_K = 60


# --------------------------------------------------------------------------- #
# Metrics (gain 2^rel - 1, trec_eval / BEIR convention)
# --------------------------------------------------------------------------- #
def ndcg(ranked, rels, k):
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = sum((2.0 ** g - 1.0) / np.log2(i + 2.0) for i, g in enumerate(ideal))
    if idcg == 0.0:
        return None
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        g = rels.get(d, 0)
        if g > 0:
            dcg += (2.0 ** g - 1.0) / np.log2(i + 2.0)
    return dcg / idcg


def mrr(ranked, rels, k):
    for i, d in enumerate(ranked[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall(ranked, rels, k):
    n_rel = sum(1 for g in rels.values() if g > 0)
    if n_rel == 0:
        return None
    hit = sum(1 for d in ranked[:k] if rels.get(d, 0) > 0)
    return hit / n_rel


# --------------------------------------------------------------------------- #
# Fusion functions. All take the two ranked lists (doc row ids, descending) and
# their raw scores, and return a fused ranking of the union.
# --------------------------------------------------------------------------- #
def _rank_maps(bm_rows, dn_rows):
    return ({int(d): r for r, d in enumerate(bm_rows)},
            {int(d): r for r, d in enumerate(dn_rows)})


def fuse_borda(bm_rows, dn_rows, bm_s, dn_s, alpha, N):
    rb, rd = _rank_maps(bm_rows, dn_rows)
    docs = list(set(rb) | set(rd))
    # points = N - rank0 (best -> N, last -> 1); missing from a list -> 0
    s = np.array([alpha * (N - rb[d] if d in rb else 0.0) +
                  (1.0 - alpha) * (N - rd[d] if d in rd else 0.0) for d in docs])
    return [docs[i] for i in np.argsort(-s, kind="stable")]


def fuse_rrf(bm_rows, dn_rows, bm_s, dn_s, alpha, N):
    """Weighted RRF: alpha/(k+rank) + (1-alpha)/(k+rank). alpha=0.5 -> plain RRF
    up to a constant factor, which does not change the ranking."""
    rb, rd = _rank_maps(bm_rows, dn_rows)
    docs = list(set(rb) | set(rd))
    s = np.array([alpha * (1.0 / (RRF_K + rb[d] + 1) if d in rb else 0.0) +
                  (1.0 - alpha) * (1.0 / (RRF_K + rd[d] + 1) if d in rd else 0.0)
                  for d in docs])
    return [docs[i] for i in np.argsort(-s, kind="stable")]


def fuse_score(bm_rows, dn_rows, bm_s, dn_s, alpha, N):
    """Convex combination of per-query MIN-MAX normalised scores. Missing from a
    list -> 0 (the bottom of that list's normalised range)."""
    def norm(rows, sc):
        sc = np.asarray(sc, dtype=np.float64)
        lo, hi = sc.min(), sc.max()
        z = (sc - lo) / (hi - lo) if hi > lo else np.zeros_like(sc)
        return {int(d): float(v) for d, v in zip(rows, z)}
    nb, nd = norm(bm_rows, bm_s), norm(dn_rows, dn_s)
    docs = list(set(nb) | set(nd))
    s = np.array([alpha * nb.get(d, 0.0) + (1.0 - alpha) * nd.get(d, 0.0) for d in docs])
    return [docs[i] for i in np.argsort(-s, kind="stable")]


FUSERS = {"borda": fuse_borda, "rrf": fuse_rrf, "score": fuse_score}


def tune_alpha(fuser, lists, qids, qrels, cid, alphas, N, k):
    """Grid-search ONE global alpha on the given split (dev)."""
    best_a, best_v = alphas[0], -1.0
    for a in alphas:
        tot, n = 0.0, 0
        for q in qids:
            rels = {d: int(g) for d, g in qrels[q].items() if int(g) > 0}
            if not rels:
                continue
            bm_r, bm_s, dn_r, dn_s = lists[q]
            ranked = [cid[i] for i in fuser(bm_r, dn_r, bm_s, dn_s, a, N)[:k]]
            v = ndcg(ranked, rels, k)
            if v is not None:
                tot += v
                n += 1
        v = tot / max(n, 1)
        if v > best_v:
            best_v, best_a = v, a
    return float(best_a), float(best_v)


# --------------------------------------------------------------------------- #
def retrieve_split(qids, queries, cid, retriever, stemmer, corpus_emb,
                   top_k, dev, dtype, chunk, qbatch, tag, cache_path):
    """BM25 + dense top-k for one split, keeping RAW SCORES (needed by RRF and
    score fusion, which the cached Borda curve cannot provide).

    Results are CACHED to disk: stage 5b (reranking) and 5c (SPLADE) reuse the
    exact same ranked lists instead of re-retrieving, and re-running this script
    costs seconds rather than ~20 minutes.
    """
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if list(z["qids"]) == list(qids):
            print(f"[bench] {tag}: reusing cached retrieval ({cache_path})")
            return {q: (z["bm_idx"][i], z["bm_val"][i], z["dn_idx"][i], z["dn_val"][i])
                    for i, q in enumerate(qids)}
        print(f"[bench] {tag}: cached qids differ -- re-retrieving")

    q_texts = [queries[q] for q in qids]
    print(f"[bench] {tag}: BM25 retrieve ({len(qids):,} queries)")
    q_tok = bm25s.tokenize(q_texts, stopwords="en", stemmer=stemmer, show_progress=False)
    bm_idx, bm_val = retriever.retrieve(q_tok, k=min(top_k, len(cid)), show_progress=True)

    print(f"[bench] {tag}: dense retrieve")
    q_emb = _MODEL.encode(q_texts, batch_size=256, show_progress_bar=True,
                          normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    dn_idx, dn_val = dense_retrieve_scores(q_emb, corpus_emb, top_k, dev, dtype, chunk, qbatch)

    np.savez(cache_path, qids=np.asarray(qids), bm_idx=np.asarray(bm_idx),
             bm_val=np.asarray(bm_val, dtype=np.float32),
             dn_idx=np.asarray(dn_idx), dn_val=np.asarray(dn_val, dtype=np.float32))
    print(f"[bench] {tag}: cached retrieval -> {cache_path}")
    return {q: (bm_idx[i], bm_val[i], dn_idx[i], dn_val[i]) for i, q in enumerate(qids)}


def main():
    config = load_config()
    paths = get_paths(config)
    name = config["dataset"]
    folder = dataset_dir(paths, name)
    seed = int(config.get("seed", 42))
    top_k = config["retrieval"]["top_k"]
    eval_k = config["retrieval"].get("eval_k", 10)
    N = config["borda"]["N"]
    n_boot = int(config["router_screen"].get("bootstrap_resamples", 1000))
    alphas = np.round(np.arange(config["borda"]["alpha_min"],
                                config["borda"]["alpha_max"] + 1e-9,
                                config["borda"]["alpha_step"]), 4)

    # ---- frozen router ----
    art = os.path.join(paths["router_final"], f"{name}_router.joblib")
    if not os.path.exists(art):
        raise SystemExit(f"[bench] {art} missing -- run final_fit.py first.")
    R = joblib.load(art)
    print(f"[bench] router: {R['family']}|{R['framing']} on {len(R['features'])} features "
          f"{R['features']}")

    # ---- corpus + retrievers ----
    pdir = processed_dir(paths, name, create=False)
    with open(os.path.join(pdir, "corpus_ids.json"), encoding="utf-8") as f:
        cid = json.load(f)
    print(f"[bench] loading corpus text ({len(cid):,} docs)")
    doc_texts = read_corpus_texts(folder, cid)
    stemmer = (Stemmer.Stemmer("english")
               if config["bm25"].get("use_stemming") and Stemmer else None)
    print("[bench] tokenising + building BM25 index")
    ctok = bm25s.tokenize(doc_texts, stopwords="en", stemmer=stemmer, show_progress=True)
    retriever = bm25s.BM25(method=config["bm25"].get("method", "lucene"),
                           k1=config["bm25"]["k1"], b=config["bm25"]["b"])
    retriever.index(ctok, show_progress=True)
    del ctok, doc_texts
    gc.collect()

    corpus_emb = np.load(os.path.join(pdir, "corpus_emb.npy"), mmap_mode="r")
    d_conf = config["dense"]
    dev_t = _resolve_device(d_conf.get("device", "cuda"))
    dtype = (torch.float16 if (dev_t == "cuda" and
             str(d_conf.get("embedding_dtype", "float32")).lower() == "float16")
             else torch.float32)
    global _MODEL
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer(d_conf["model_name"], device=dev_t)
    if d_conf.get("max_seq_length"):
        _MODEL.max_seq_length = d_conf["max_seq_length"]

    queries = read_queries(folder)
    chunk = int(d_conf.get("retrieval_chunk_size", 50000))
    qbatch = int(d_conf.get("query_batch_size", 2048))

    # ---- retrieve dev (for tuning the global-alpha baselines) and test ----
    out = {}
    for split in ("dev", "test"):
        qr = read_qrels(folder, split)
        if qr is None:
            raise SystemExit(f"[bench] no qrels for split '{split}'.")
        qids = [q for q in queries if q in qr and len(qr[q]) > 0]
        cache = os.path.join(pdir, f"retrieval_{split}_top{top_k}.npz")
        out[split] = (qids, qr, retrieve_split(qids, queries, cid, retriever, stemmer,
                                               corpus_emb, top_k, dev_t, dtype, chunk,
                                               qbatch, split, cache))
    dev_q, dev_qrels, dev_lists = out["dev"]
    test_q, test_qrels, test_lists = out["test"]

    # ---- tune the global alphas on DEV (never on test) ----
    print("\n[bench] tuning global alphas on dev ...")
    a_borda, v_b = tune_alpha(fuse_borda, dev_lists, dev_q, dev_qrels, cid, alphas, N, eval_k)
    a_rrf, v_r = tune_alpha(fuse_rrf, dev_lists, dev_q, dev_qrels, cid, alphas, N, eval_k)
    a_scr, v_s = tune_alpha(fuse_score, dev_lists, dev_q, dev_qrels, cid, alphas, N, eval_k)
    print(f"[bench]   Borda alpha*={a_borda:.2f} (dev {v_b:.4f}) | "
          f"wRRF alpha*={a_rrf:.2f} (dev {v_r:.4f}) | score alpha*={a_scr:.2f} (dev {v_s:.4f})")

    # ---- router alphas on test (features already computed by create_dataset) ----
    fcsv = os.path.join(paths["feature_dataset"], f"{name}_test_features.csv")
    tdf = pd.read_csv(fcsv).set_index("qid")
    Xte = tdf.loc[[q for q in test_q if q in tdf.index], R["features"]].to_numpy(dtype=np.float64)
    feat_qids = [q for q in test_q if q in tdf.index]
    t_pred = time.perf_counter()
    raw = predict_alpha(R["model"], Xte, R["framing"], R["bins"])
    if R["decision"] == "calibrated" and R["calib_edges"] is not None:
        raw = apply_calibration(raw, R["calib_edges"], R["calib_bin_alpha"])
    router_secs = time.perf_counter() - t_pred
    router_alpha = {q: float(a) for q, a in zip(feat_qids, raw)}
    print(f"[bench] router predicted {len(router_alpha):,} alphas in {router_secs*1000:.1f} ms "
          f"({router_secs/max(len(router_alpha),1)*1e6:.1f} us/query) "
          f"mean={raw.mean():.3f} std={raw.std():.3f}")

    # ---- evaluate every method on TEST ----
    METHODS = [
        ("BM25",                    "bm25",  None),
        ("Dense (all-mpnet)",       "dense", None),
        ("Static Borda a=0.5",      "borda", 0.5),
        (f"Static Borda a*={a_borda:.2f} [BASELINE]", "borda", a_borda),
        ("RRF k=60",                "rrf",   0.5),
        (f"Weighted RRF a*={a_rrf:.2f}",  "rrf",   a_rrf),
        (f"Norm-score fusion a*={a_scr:.2f}", "score", a_scr),
        ("ROUTER (ours)",           "router", None),
        ("Oracle alpha",            "oracle", None),
    ]
    per_q = {m[0]: {"ndcg10": [], "ndcg100": [], "mrr100": [], "recall100": []}
             for m in METHODS}
    used = []
    print(f"\n[bench] scoring {len(test_q):,} test queries x {len(METHODS)} methods ...")
    for n_done, q in enumerate(test_q, 1):
        rels = {d: int(g) for d, g in test_qrels[q].items() if int(g) > 0}
        if not rels:
            continue
        bm_r, bm_s, dn_r, dn_s = test_lists[q]
        used.append(q)
        for label, kind, a in METHODS:
            if kind == "bm25":
                ranked = [cid[int(i)] for i in bm_r]
            elif kind == "dense":
                ranked = [cid[int(i)] for i in dn_r]
            elif kind == "oracle":
                best, ranked = -1.0, None
                for aa in alphas:
                    cand = [cid[i] for i in fuse_borda(bm_r, dn_r, bm_s, dn_s, aa, N)[:eval_k]]
                    v = ndcg(cand, rels, eval_k)
                    if v is not None and v > best:
                        best, ranked = v, cand
                ranked = ranked or []
            else:
                aa = router_alpha.get(q, a_borda) if kind == "router" else a
                f = fuse_borda if kind in ("router", "borda") else FUSERS[kind]
                ranked = [cid[i] for i in f(bm_r, dn_r, bm_s, dn_s, aa, N)]
            r = per_q[label]
            r["ndcg10"].append(ndcg(ranked, rels, eval_k) or 0.0)
            r["ndcg100"].append(ndcg(ranked, rels, 100) or 0.0)
            r["mrr100"].append(mrr(ranked, rels, 100))
            r["recall100"].append(recall(ranked, rels, 100) or 0.0)
        if n_done % 500 == 0:
            print(f"\r[bench]   {n_done:,}/{len(test_q):,}", end="", flush=True)
    print(f"\r[bench]   {len(used):,} queries scored.          ")

    # ---- table + significance vs the primary baseline ----
    base_label = [m[0] for m in METHODS if "[BASELINE]" in m[0]][0]
    base = np.asarray(per_q[base_label]["ndcg10"])
    rows = []
    for label, _, _ in METHODS:
        d = {k: np.asarray(v) for k, v in per_q[label].items()}
        lo, hi = bootstrap_ci(d["ndcg10"], n_boot, seed)
        diff, dlo, dhi = paired_bootstrap(d["ndcg10"], base, n_boot, seed)
        rows.append(dict(method=label,
                         ndcg10=d["ndcg10"].mean(), ci_lo=lo, ci_hi=hi,
                         ndcg100=d["ndcg100"].mean(), mrr100=d["mrr100"].mean(),
                         recall100=d["recall100"].mean(),
                         diff_vs_baseline=diff, diff_ci_lo=dlo, diff_ci_hi=dhi,
                         significant=bool(dlo > 0 or dhi < 0)))
    df = pd.DataFrame(rows)

    o1 = os.path.join(paths["router_final"], f"{name}_benchmark.csv")
    o2 = os.path.join(paths["router_final"], f"{name}_benchmark_per_query.csv")
    o3 = os.path.join(paths["router_final"], f"{name}_benchmark.json")
    df.to_csv(o1, index=False)
    pd.DataFrame({"qid": used, **{f"{l}_ndcg10": per_q[l]["ndcg10"] for l, _, _ in METHODS}}
                 ).to_csv(o2, index=False)
    router_row = df[df.method == "ROUTER (ours)"].iloc[0]
    with open(o3, "w", encoding="utf-8") as f:
        json.dump(dict(dataset=name, split="test", n_queries=len(used), eval_k=eval_k,
                       baseline=base_label, alpha_borda=a_borda, alpha_rrf=a_rrf,
                       alpha_score=a_scr,
                       router_ndcg10=float(router_row["ndcg10"]),
                       router_gain=float(router_row["diff_vs_baseline"]),
                       router_gain_ci=[float(router_row["diff_ci_lo"]),
                                       float(router_row["diff_ci_hi"])],
                       router_significant=bool(router_row["significant"]),
                       router_us_per_query=router_secs / max(len(router_alpha), 1) * 1e6,
                       table=df.to_dict("records")), f, indent=2)

    pd.set_option("display.width", 200)
    print(f"\n[bench] wrote {o1}\n[bench] wrote {o2}\n[bench] wrote {o3}\n")
    print(df[["method", "ndcg10", "ci_lo", "ci_hi", "ndcg100", "mrr100", "recall100",
              "diff_vs_baseline", "significant"]].to_string(index=False))
    print(f"\n[bench] baseline = {base_label}")
    print(f"[bench] ROUTER vs baseline: {router_row['diff_vs_baseline']:+.4f} "
          f"CI [{router_row['diff_ci_lo']:+.4f}, {router_row['diff_ci_hi']:+.4f}] "
          f"{'SIGNIFICANT' if router_row['significant'] else 'NOT significant'}")
    print("[bench] TEST IS NOW SPENT -- do not tune anything against these numbers.")


if __name__ == "__main__":
    main()
