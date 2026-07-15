"""create_dataset.py -- build the Phase-2 feature dataset for the active dataset.

Produces ONE training table per split (train/dev/test): one row per query,
columns = inference-time features (see docs/feature_dataset.md), label = the
ORACLE ALPHA, recomputed fresh here at retrieval.top_k / retrieval.eval_k so
that features and label come from the exact same retrieval configuration.

Design (see docs/feature_dataset.md for the full rationale):
  * Corpus-level assets (term-doc count matrix, df/cf/idf/VAR tables, collection
    centroid) are built ONCE and cached, then reused for every split.
  * Query embeddings are computed IN-SCRIPT per split (no per-split embed.py run).
  * Per-retriever score-distribution features are computed on BOTH the BM25 and
    the dense list (`*_bm25` / `*_dense`).
  * Cross-retriever comparisons use Z-SCORE-normalised score vectors (each
    retriever's top-k scores standardised to mean 0 / std 1 per query) so BM25
    scores and cosine similarities are put on a common scale before differencing.
  * Every feature group is toggleable from config (`create_dataset.features`)
    for the planned ablation study.

Prerequisite: embed.py must have produced corpus_emb.npy / corpus_ids.json for
the active dataset (the corpus embeddings are reused as-is).

Per-split output: data/results/feature_dataset/<dataset>_<split>_features.csv
Re-running skips any split whose CSV already exists (resumable).
"""

import os
import sys
import gc
import json
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.stats import kendalltau

import torch
import bm25s
try:
    import Stemmer
except ImportError:
    Stemmer = None

from utils import load_config, get_paths, dataset_dir, processed_dir, build_doc_text
from alpha_distribution import ndcg_at_k, oracle_alpha


# =========================================================================== #
# Direct BEIR readers (load corpus ONCE; queries/qrels per split, no reload)
# =========================================================================== #
def read_corpus_texts(folder, canonical_ids):
    """Return doc texts ordered to match `canonical_ids` (corpus_emb row order)."""
    text_by_id = {}
    with open(os.path.join(folder, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text_by_id[d["_id"]] = build_doc_text(d)
    return [text_by_id.get(i, "") for i in canonical_ids]


def read_queries(folder):
    q = {}
    with open(os.path.join(folder, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            q[d["_id"]] = d.get("text", "")
    return q


def read_qrels(folder, split):
    """Return {qid: {doc_id: rel}} for a split, or None if the split file is absent."""
    path = os.path.join(folder, "qrels", f"{split}.tsv")
    if not os.path.exists(path):
        return None
    qrels = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()  # query-id \t corpus-id \t score
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, rel = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[did] = rel
    return qrels


# =========================================================================== #
# One-time corpus assets (cached under processed_data/<dataset>/assets/)
# =========================================================================== #
def build_corpus_assets(paths, name, doc_texts, bm25_conf, keep_token_ids):
    """Build (or load) term stats + inverted index + centroid. Also returns the
    fitted bm25s retriever and the vocab. `keep_token_ids` keeps per-doc token
    ids in RAM (needed only for Clarity Score)."""
    adir = os.path.join(processed_dir(paths, name), "assets")
    os.makedirs(adir, exist_ok=True)
    npz_path = os.path.join(adir, "termdoc.npz")
    stat_path = os.path.join(adir, "termstats.npz")
    vocab_path = os.path.join(adir, "vocab.json")

    stemmer = None
    if bm25_conf.get("use_stemming", False) and Stemmer is not None:
        stemmer = Stemmer.Stemmer("english")

    print(f"[dataset] tokenising {len(doc_texts):,} docs + building BM25 index")
    corpus_tokens = bm25s.tokenize(doc_texts, stopwords="en", stemmer=stemmer, show_progress=True)
    retriever = bm25s.BM25(method=bm25_conf.get("method", "lucene"),
                           k1=bm25_conf["k1"], b=bm25_conf["b"])
    retriever.index(corpus_tokens, show_progress=True)

    ids = corpus_tokens.ids            # list of per-doc token-id arrays
    vocab = corpus_tokens.vocab        # dict token(str) -> id
    n_doc = len(ids)

    if os.path.exists(npz_path) and os.path.exists(stat_path) and os.path.exists(vocab_path):
        print("[dataset] loading cached corpus assets")
        M = load_npz(npz_path)
        stats = np.load(stat_path)
        df, cf, idf, var_t = stats["df"], stats["cf"], stats["idf"], stats["var_t"]
        tokens_coll = int(stats["tokens_coll"])
    else:
        print("[dataset] building term-document count matrix (one-time)")
        rows, cols, data = [], [], []
        for d, arr in enumerate(tqdm(ids, desc="[dataset] inverted index")):
            arr = np.asarray(arr)
            if arr.size == 0:
                continue
            u, c = np.unique(arr, return_counts=True)
            rows.append(u); cols.append(np.full(u.shape, d, dtype=np.int64)); data.append(c)
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        data = np.concatenate(data).astype(np.float64)
        V = len(vocab)
        M = csr_matrix((data, (rows, cols)), shape=(V, n_doc))   # term x doc counts
        df = np.diff(M.indptr).astype(np.float64)                # doc freq per term
        cf = np.asarray(M.sum(axis=1)).ravel()                   # collection freq per term
        tokens_coll = int(cf.sum())
        idf = np.log(n_doc / np.maximum(df, 1.0))                # idf(t) = log(N/df)
        # VAR(t): variance of w_{t,d} = (1+ln tf)*idf(t) across docs containing t
        var_t = np.zeros(V, dtype=np.float64)
        for t in tqdm(range(V), desc="[dataset] term-weight variance"):
            a, b = M.indptr[t], M.indptr[t + 1]
            if b - a < 1:
                continue
            w = (1.0 + np.log(M.data[a:b])) * idf[t]
            var_t[t] = w.var()
        save_npz(npz_path, M)
        np.savez(stat_path, df=df, cf=cf, idf=idf, var_t=var_t,
                 tokens_coll=np.int64(tokens_coll))
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f)

    avgdl = tokens_coll / max(n_doc, 1)

    # collection embedding centroid (streamed to stay memmap-friendly)
    cen_path = os.path.join(adir, "centroid.npy")
    corpus_emb = np.load(os.path.join(processed_dir(paths, name), "corpus_emb.npy"), mmap_mode="r")
    if os.path.exists(cen_path):
        centroid = np.load(cen_path)
    else:
        print("[dataset] computing collection embedding centroid")
        acc = np.zeros(corpus_emb.shape[1], dtype=np.float64)
        step = 100000
        for s in tqdm(range(0, corpus_emb.shape[0], step), desc="[dataset] centroid"):
            acc += np.asarray(corpus_emb[s:s + step], dtype=np.float64).sum(axis=0)
        centroid = (acc / corpus_emb.shape[0]).astype(np.float32)
        np.save(cen_path, centroid)

    assets = dict(M=M, df=df, cf=cf, idf=idf, var_t=var_t, vocab=vocab,
                  tokens_coll=tokens_coll, n_doc=n_doc, avgdl=avgdl,
                  centroid=centroid, corpus_emb=corpus_emb,
                  token_ids=(ids if keep_token_ids else None))
    return retriever, stemmer, assets


# =========================================================================== #
# Small math helpers
# =========================================================================== #
def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _bm25_corpus_baseline(term_ids, assets, k1, b):
    """BM25 score of the query against the whole corpus treated as one document."""
    N, avgdl = assets["n_doc"], assets["avgdl"]
    df, cf, tokens_coll = assets["df"], assets["cf"], assets["tokens_coll"]
    total = 0.0
    for t in term_ids:
        tf = cf[t]                                             # term count in the mega-doc
        idf_l = math.log(1.0 + (N - df[t] + 0.5) / (df[t] + 0.5))   # lucene idf
        total += idf_l * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * tokens_coll / avgdl))
    return total


# =========================================================================== #
# Feature blocks
# =========================================================================== #
def query_only_features(term_ids, qtf, ql, q_emb, assets, feat_cfg, eps):
    """Group A: query-only features (all from cached term stats + query embedding)."""
    f = {}
    df, cf, idf_all, var_all = assets["df"], assets["cf"], assets["idf"], assets["var_t"]
    tokens_coll, N = assets["tokens_coll"], assets["n_doc"]

    f["ql"] = float(ql)
    if term_ids:
        idfs = idf_all[term_ids]
        f["avg_idf"] = float(idfs.mean())
        f["max_idf"] = float(idfs.max())
        f["std_idf"] = float(idfs.std())
        f["idf_ratio"] = float(idfs.max() / (idfs.min() + eps))
        # SCS: sum P_ml(w|Q) log2(P_ml/P_coll)
        pml = qtf / ql
        pcoll = cf[term_ids] / tokens_coll
        f["scs"] = float((pml * np.log2((pml + eps) / (pcoll + eps))).sum())
        f["avictf"] = float(np.log2(tokens_coll / (cf[term_ids] + eps)).mean())
        scq = (1.0 + np.log(cf[term_ids] + eps)) * np.log(1.0 + N / (df[term_ids] + eps))
        f["scq_sum"], f["scq_avg"], f["scq_max"] = float(scq.sum()), float(scq.mean()), float(scq.max())
        vt = var_all[term_ids]
        f["var_sum"], f["var_avg"], f["var_max"] = float(vt.sum()), float(vt.mean()), float(vt.max())
    else:  # all query terms OOV
        for kk in ("avg_idf", "max_idf", "std_idf", "idf_ratio", "scs", "avictf",
                   "scq_sum", "scq_avg", "scq_max", "var_sum", "var_avg", "var_max"):
            f[kk] = 0.0

    # embedding-based query specificity: cosine(query, collection centroid)
    cen = assets["centroid"]
    denom = (np.linalg.norm(q_emb) * np.linalg.norm(cen)) + eps
    f["query_centroid_cos"] = float(np.dot(q_emb, cen) / denom)

    if feat_cfg.get("query_scope", True):
        f["query_scope"] = _query_scope(term_ids, assets)
    if feat_cfg.get("pmi", True):
        f["pmi_avg"] = _pmi_avg(term_ids, assets, eps)
    return f


def _postings(t, M):
    return M.indices[M.indptr[t]:M.indptr[t + 1]]


def _query_scope(term_ids, assets):
    if not term_ids:
        return 0.0
    M, N = assets["M"], assets["n_doc"]
    union = np.unique(np.concatenate([_postings(t, M) for t in term_ids])) if term_ids else np.array([])
    n_q = max(len(union), 1)
    return float(-math.log(n_q / N))


def _pmi_avg(term_ids, assets, eps):
    uniq = list(dict.fromkeys(term_ids))
    if len(uniq) < 2:
        return 0.0
    M, N = assets["M"], assets["n_doc"]
    df = assets["df"]
    vals = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            ti, tj = uniq[i], uniq[j]
            co = np.intersect1d(_postings(ti, M), _postings(tj, M), assume_unique=False).size
            p_ij = (co + eps) / N
            p_i, p_j = (df[ti] + eps) / N, (df[tj] + eps) / N
            vals.append(math.log(p_ij / (p_i * p_j)))
    return float(np.mean(vals)) if vals else 0.0


def per_retriever_features(prefix, scores, rows, baseline, assets, feat_cfg, cfg, eps):
    """Groups B/C for ONE retriever's ranked list. `scores` desc, `rows` = doc row ids."""
    f = {}
    s = np.asarray(scores, dtype=np.float64)
    k = len(s)
    mu, sd = s.mean(), s.std()
    f[f"top_score_{prefix}"] = float(s[0])
    f[f"sigma_k_{prefix}"] = float(sd)
    f[f"margin_{prefix}"] = float(s[0] - s[1]) if k > 1 else 0.0
    f[f"norm_margin_{prefix}"] = float((s[0] - s[1]) / (abs(s[0]) + eps)) if k > 1 else 0.0
    f[f"wig_{prefix}"] = float(mu - baseline)
    f[f"nqc_{prefix}"] = float(sd / (abs(baseline) + eps))
    sp = np.maximum(s, eps)                                  # SMV assumes positive magnitudes
    mup = sp.mean()
    f[f"smv_{prefix}"] = float(((sp / mup) * np.abs(np.log(sp / mup))).mean())
    f[f"entropy_{prefix}"] = _entropy(_softmax(s))
    trim = max(1, int(0.1 * k))
    f[f"robust_sigma_{prefix}"] = float(np.sort(s)[trim:k - trim].std()) if k - 2 * trim > 1 else float(sd)

    if feat_cfg.get("coherence", True):
        f.update(_coherence(prefix, s, rows, assets, cfg, eps))
    return f


def _coherence(prefix, s, rows, assets, cfg, eps):
    """Score autocorrelation (Moran's I) + A-Pair-Ratio over a top-W / bottom-W window."""
    W = int(cfg.get("coherence_window", 100))
    emb = assets["corpus_emb"]
    k = len(rows)
    w = min(W, k)
    top_rows = np.asarray(rows[:w])
    E = np.asarray(emb[top_rows], dtype=np.float32)          # (w, dim), already L2-normalised
    sim = E @ E.T
    np.fill_diagonal(sim, 0.0)

    # Moran's I of the top-window scores over the doc-similarity graph
    z = s[:w] - s[:w].mean()
    S0 = sim.sum()
    den = (z * z).sum()
    moran = float((w / (S0 + eps)) * (z @ sim @ z) / (den + eps)) if S0 > 0 and den > 0 else 0.0

    # A-Pair-Ratio: mean pairwise cos(top-W) / mean pairwise cos(bottom-W)
    def _mean_pair(sm):
        n = sm.shape[0]
        if n < 2:
            return 0.0
        return float(sm.sum() / (n * (n - 1)))              # diag already 0
    top_mean = _mean_pair(sim)
    if k >= 2 * w:
        Eb = np.asarray(emb[np.asarray(rows[-w:])], dtype=np.float32)
        simb = Eb @ Eb.T
        np.fill_diagonal(simb, 0.0)
        bot_mean = _mean_pair(simb)
    else:
        bot_mean = top_mean
    return {f"autocorr_{prefix}": moran,
            f"apair_ratio_{prefix}": float(top_mean / (bot_mean + eps))}


def clarity_feature(prefix, scores, rows, assets, cfg, eps):
    """Clarity Score for ONE retriever, over a capped feedback set of top docs."""
    ids = assets["token_ids"]
    if ids is None:
        return {f"clarity_{prefix}": 0.0}
    fb = int(cfg.get("clarity_feedback_k", 50))
    fb = min(fb, len(rows))
    weights = _softmax(np.asarray(scores[:fb], dtype=np.float64))   # P(d|Q)
    agg = {}
    for r, wgt in zip(rows[:fb], weights):
        arr = np.asarray(ids[int(r)])
        if arr.size == 0:
            continue
        u, c = np.unique(arr, return_counts=True)
        p_doc = c / arr.size
        for tid, p in zip(u, p_doc):
            agg[int(tid)] = agg.get(int(tid), 0.0) + wgt * p
    cf, tokens_coll = assets["cf"], assets["tokens_coll"]
    cs = 0.0
    for tid, pwq in agg.items():
        pcoll = cf[tid] / tokens_coll
        if pwq > 0 and pcoll > 0:
            cs += pwq * math.log2(pwq / pcoll)
    return {f"clarity_{prefix}": float(cs)}


def cross_retriever_features(bm_rows, dn_rows, s_bm, s_dn, base_bm, base_dn, eps):
    """Group D: rank-list agreement + Z-SCORE-normalised score-difference features."""
    f = {}
    set_b, set_d = set(bm_rows.tolist()), set(dn_rows.tolist())
    inter = set_b & set_d
    union = set_b | set_d
    f["jaccard"] = float(len(inter) / len(union)) if union else 0.0

    # Kendall's tau over docs present in BOTH lists (by rank position)
    if len(inter) >= 2:
        rank_b = {d: i for i, d in enumerate(bm_rows.tolist())}
        rank_d = {d: i for i, d in enumerate(dn_rows.tolist())}
        common = list(inter)
        tau, _ = kendalltau([rank_b[d] for d in common], [rank_d[d] for d in common])
        f["kendall_tau"] = float(tau) if tau == tau else 0.0     # NaN -> 0
    else:
        f["kendall_tau"] = 0.0

    # Z-score each retriever's top-k scores, then difference the standardised stats.
    def _z(s):
        s = np.asarray(s, dtype=np.float64)
        mu, sd = s.mean(), s.std()
        z = (s - mu) / (sd + eps)
        zt = z[0]
        zm = (s[0] - s[1]) / (sd + eps) if len(s) > 1 else 0.0
        zent = _entropy(_softmax(z))
        return zt, zm, zent, sd, mu
    zt_b, zm_b, zent_b, sd_b, mu_b = _z(s_bm)
    zt_d, zm_d, zent_d, sd_d, mu_d = _z(s_dn)
    f["d_ztop"] = float(zt_b - zt_d)
    f["d_zmargin"] = float(zm_b - zm_d)
    f["d_zentropy"] = float(zent_b - zent_d)
    f["d_wig_z"] = float((mu_b - base_bm) / (sd_b + eps) - (mu_d - base_dn) / (sd_d + eps))
    return f


# =========================================================================== #
# Dense retrieval WITH scores (query-batched; preloads corpus to GPU if it fits)
# =========================================================================== #
def dense_retrieve_scores(q_emb, corpus_emb, top_k, dev, dtype, chunk, qbatch=2048):
    n_doc, dim = corpus_emb.shape
    k = min(top_k, n_doc)
    nq = q_emb.shape[0]
    out_idx = np.empty((nq, k), dtype=np.int64)
    out_val = np.empty((nq, k), dtype=np.float32)

    bytes_per = 2 if dtype == torch.float16 else 4
    Cgpu = None
    if dev == "cuda" and n_doc * dim * bytes_per < 18e9:        # fits comfortably in 24 GB
        Cgpu = torch.from_numpy(np.ascontiguousarray(corpus_emb[:])).to(dev).to(dtype)

    for qs in tqdm(range(0, nq, qbatch), desc="[dataset] dense retrieve"):
        qe = min(qs + qbatch, nq)
        Q = torch.from_numpy(np.ascontiguousarray(q_emb[qs:qe])).to(dev).to(dtype)
        if Cgpu is not None:
            sims = Q @ Cgpu.T
            vals, idx = torch.topk(sims, k, dim=1)
            out_idx[qs:qe], out_val[qs:qe] = idx.cpu().numpy(), vals.float().cpu().numpy()
            del sims, vals, idx
            continue
        run_v = run_i = None
        for start in range(0, n_doc, chunk):
            end = min(start + chunk, n_doc)
            block = torch.from_numpy(np.ascontiguousarray(corpus_emb[start:end])).to(dev).to(dtype)
            sims = Q @ block.T
            kk = min(k, end - start)
            vals, idx = torch.topk(sims, kk, dim=1)
            idx = idx + start
            if run_v is None:
                run_v, run_i = vals, idx
            else:
                run_v = torch.cat([run_v, vals], 1); run_i = torch.cat([run_i, idx], 1)
                vals, sel = torch.topk(run_v, min(k, run_v.shape[1]), 1)
                run_v, run_i = vals, torch.gather(run_i, 1, sel)
            del block, sims
        out_idx[qs:qe], out_val[qs:qe] = run_i.cpu().numpy(), run_v.float().cpu().numpy()
    del Cgpu
    return out_idx, out_val


# =========================================================================== #
def _resolve_device(requested):
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    print("[dataset] CUDA not available -- using CPU.")
    return "cpu"


def process_split(split, folder, canonical_ids, retriever, stemmer, assets,
                  model, config, paths, name, dev, dtype):
    out_csv = os.path.join(paths["feature_dataset"], f"{name}_{split}_features.csv")
    if os.path.exists(out_csv):
        print(f"[dataset] split '{split}' already built ({out_csv}) -- skipping.")
        return
    qrels = read_qrels(folder, split)
    if qrels is None:
        print(f"[dataset] split '{split}' has no qrels file -- skipping.")
        return

    queries = read_queries(folder)
    qids = [q for q in queries if q in qrels and len(qrels[q]) > 0]
    cds = config["create_dataset"]
    cap = cds.get("debug_max_queries")
    if cap:
        qids = qids[:int(cap)]
    if not qids:
        print(f"[dataset] split '{split}': no queries with qrels -- skipping.")
        return

    feat_cfg = cds.get("features", {})
    eps = float(cds.get("zscore_eps", 1e-9))
    top_k = config["retrieval"]["top_k"]
    eval_k = config["retrieval"].get("eval_k", 10)
    N = config["borda"]["N"]
    k1, b = config["bm25"]["k1"], config["bm25"]["b"]
    alphas = np.round(np.arange(config["borda"]["alpha_min"],
                                config["borda"]["alpha_max"] + 1e-9,
                                config["borda"]["alpha_step"]), 4)

    print(f"[dataset] split '{split}': {len(qids):,} queries -> embedding + retrieving")
    q_texts = [queries[q] for q in qids]

    # dense: embed this split's queries in-script, retrieve with scores
    q_emb = model.encode(q_texts, batch_size=config["dense"].get("batch_size", 256),
                         show_progress_bar=True, normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
    dn_idx, dn_val = dense_retrieve_scores(q_emb, assets["corpus_emb"], top_k, dev, dtype,
                                           int(config["dense"].get("retrieval_chunk_size", 50000)))

    # bm25: tokenise queries (strings) and retrieve with scores
    q_tokens = bm25s.tokenize(q_texts, stopwords="en", stemmer=stemmer, show_progress=False)
    bm_idx, bm_val = retriever.retrieve(q_tokens, k=min(top_k, assets["n_doc"]), show_progress=True)

    # query term ids mapped to the corpus vocab (for group-A features)
    vocab = assets["vocab"]
    q_tok_str = bm25s.tokenize(q_texts, stopwords="en", stemmer=stemmer,
                               return_ids=False, show_progress=False)

    rows = []
    for i, q in enumerate(tqdm(qids, desc=f"[dataset] features '{split}'")):
        rels = {d: int(g) for d, g in qrels[q].items() if int(g) > 0}
        if not rels:
            continue
        bm_r, dn_r = bm_idx[i], dn_idx[i]
        s_bm, s_dn = np.asarray(bm_val[i], dtype=np.float64), np.asarray(dn_val[i], dtype=np.float64)
        bm_docs = [canonical_ids[int(j)] for j in bm_r]
        dn_docs = [canonical_ids[int(j)] for j in dn_r]

        a_star, nd = oracle_alpha(bm_docs, dn_docs, rels, N, alphas, eval_k)
        if a_star is None:
            continue

        row = {"dataset": name, "split": split, "qid": q}

        if feat_cfg.get("query_only", True):
            toks = [vocab[t] for t in q_tok_str[i] if t in vocab]
            uniq, counts = (np.unique(toks, return_counts=True) if toks else (np.array([], int), np.array([])))
            ql = len(q_tok_str[i])
            row.update(query_only_features(list(uniq), counts.astype(float), max(ql, 1),
                                           q_emb[i], assets, feat_cfg, eps))

        base_bm = _bm25_corpus_baseline([vocab[t] for t in q_tok_str[i] if t in vocab], assets, k1, b)
        base_dn = float(np.dot(q_emb[i], assets["centroid"]))
        if feat_cfg.get("per_retriever_scores", True):
            row.update(per_retriever_features("bm25", s_bm, bm_r, base_bm, assets, feat_cfg, cds, eps))
            row.update(per_retriever_features("dense", s_dn, dn_r, base_dn, assets, feat_cfg, cds, eps))
        if feat_cfg.get("clarity", True):
            row.update(clarity_feature("bm25", s_bm, bm_r, assets, cds, eps))
            row.update(clarity_feature("dense", s_dn, dn_r, assets, cds, eps))
        if feat_cfg.get("cross_retriever", True):
            row.update(cross_retriever_features(bm_r, dn_r, s_bm, s_dn, base_bm, base_dn, eps))

        # label + references
        row.update({"alpha": a_star, "oracle_ndcg": nd,
                    "bm25_ndcg": ndcg_at_k(bm_docs, rels, eval_k) or 0.0,
                    "dense_ndcg": ndcg_at_k(dn_docs, rels, eval_k) or 0.0,
                    "n_rel": len(rels), "eval_k": eval_k, "top_k": top_k})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[dataset] wrote {out_csv}  ({len(df):,} rows x {df.shape[1]} cols)")
    del q_emb, dn_idx, dn_val, bm_idx, bm_val
    gc.collect()


def main():
    config = load_config()
    paths = get_paths(config)
    name = config["dataset"]
    folder = dataset_dir(paths, name)
    cds = config["create_dataset"]

    pdir = processed_dir(paths, name, create=False)
    cid_path = os.path.join(pdir, "corpus_ids.json")
    if not os.path.exists(cid_path):
        raise SystemExit(f"[dataset] {cid_path} missing -- run embed.py for '{name}' first "
                         f"(corpus embeddings are reused as-is).")
    with open(cid_path, encoding="utf-8") as f:
        canonical_ids = json.load(f)

    print(f"[dataset] '{name}': loading corpus text ({len(canonical_ids):,} docs)")
    doc_texts = read_corpus_texts(folder, canonical_ids)

    keep_ids = bool(cds.get("features", {}).get("clarity", True))
    retriever, stemmer, assets = build_corpus_assets(
        paths, name, doc_texts, config["bm25"], keep_token_ids=keep_ids)
    del doc_texts
    gc.collect()

    dev = _resolve_device(config["dense"].get("device", "cuda"))
    stored = str(config["dense"].get("embedding_dtype", "float32")).lower()
    dtype = torch.float16 if (dev == "cuda" and stored == "float16") else torch.float32

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config["dense"]["model_name"], device=dev)
    if config["dense"].get("max_seq_length"):
        model.max_seq_length = config["dense"]["max_seq_length"]

    for split in cds.get("splits", ["train", "dev", "test"]):
        process_split(split, folder, canonical_ids, retriever, stemmer, assets,
                      model, config, paths, name, dev, dtype)

    print("[dataset] done.")


if __name__ == "__main__":
    main()
