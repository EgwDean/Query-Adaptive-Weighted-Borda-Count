"""pipeline.py -- Query-Adaptive Score Fusion: the entire pipeline, one command.

    python src/pipeline.py                 # run everything, skipping finished sections
    python src/pipeline.py --from 4        # force re-run from section 4 onward
    python src/pipeline.py --only 5        # run just one section

METHOD
------
Two retrievers (tuned BM25, all-mpnet-base-v2) are fused by a CONVEX COMBINATION
of per-query min-max normalised scores:

    fuse(d) = alpha * norm(bm25_score(d)) + (1-alpha) * norm(dense_score(d))

alpha = 1 -> pure lexical, alpha = 0 -> pure semantic. Convex score fusion is
used (rather than rank fusion like RRF/Borda) because rank-based fusion discards
score MAGNITUDE -- the information that says "doc A and B are both excellent,
C is junk" -- keeping only position. See Bruch, Gai & Ingber, "An Analysis of
Fusion Functions for Hybrid Retrieval", ACM TOIS 2023.

CLAIM UNDER TEST: a per-query alpha, predicted by a cheap router, beats the best
single GLOBAL alpha. Rank-fusion methods (RRF, Borda) appear only as standard
baselines, not as a contribution.

SECTIONS (each writes files; re-running skips any section whose outputs exist)
------------------------------------------------------------------------------
  0 download   BEIR dataset
  1 embed      corpus + query embeddings (memmap, sharded)
  2 tune_bm25  grid-search k1/b/stemming by NDCG@eval_k
  3 retrieve   BM25 + dense top-k WITH RAW SCORES, cached per split
  4 dataset    router features + alpha->NDCG curve + oracle alpha label
  5 screen     model families x framings x decision rules  (Optuna, on dev)
  6 ablate     greedy backward feature elimination
  7 rescreen   families x framings x feature sets
  8 final_fit  refit the winner on the full train split and FREEZE it
  9 benchmark  all baselines vs the router on TEST -- run once

Section 3 caches the ranked lists and raw scores separately from the fusion, so
changing the fusion function only re-runs section 4 (minutes), not retrieval
(hours).
"""

import os
import sys
import gc
import json
import time
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tqdm import tqdm

from utils import load_config, get_paths, dataset_dir, processed_dir, build_doc_text
from core import (N_THREADS, RRF_K, FUSERS, read_corpus_texts, read_queries,
                  read_qrels, load_retrieval, ndcg)

warnings.filterwarnings("ignore")


# =========================================================================== #
# SECTION 0-1: download + embed
# =========================================================================== #
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{n}.zip"


def sec_download(cfg, paths):
    name = cfg["dataset"]
    tgt = dataset_dir(paths, name)
    if os.path.exists(os.path.join(tgt, "corpus.jsonl")):
        return f"already present: {tgt}"
    from beir import util
    util.download_and_unzip(BEIR_URL.format(n=name), paths["datasets"])
    return f"downloaded -> {tgt}"


def sec_embed(cfg, paths):
    from numpy.lib.format import open_memmap
    from sentence_transformers import SentenceTransformer
    import torch
    name, d = cfg["dataset"], cfg["dense"]
    out = processed_dir(paths, name)
    emb_p = os.path.join(out, "corpus_emb.npy")
    ids_p = os.path.join(out, "corpus_ids.json")
    if os.path.exists(emb_p) and os.path.exists(ids_p):
        a = np.load(emb_p, mmap_mode="r")
        return f"already present: {a.shape} {a.dtype}"

    folder = dataset_dir(paths, name)
    doc_ids, texts = [], []
    with open(os.path.join(folder, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            doc_ids.append(j["_id"])
            texts.append(build_doc_text(j))

    dev = "cuda" if (d.get("device", "cuda") != "cpu" and torch.cuda.is_available()) else "cpu"
    model = SentenceTransformer(d["model_name"], device=dev)
    if d.get("max_seq_length"):
        model.max_seq_length = d["max_seq_length"]
    dt = np.float16 if str(d.get("embedding_dtype", "float32")).lower() == "float16" else np.float32
    dim = model.get_sentence_embedding_dimension()
    shard = int(d.get("encode_shard_size", 100000))

    mm = open_memmap(emb_p, mode="w+", dtype=dt, shape=(len(doc_ids), dim))
    for s in tqdm(range(0, len(doc_ids), shard), desc="  corpus shards", unit="shard"):
        e = min(s + shard, len(doc_ids))
        mm[s:e] = model.encode(texts[s:e], batch_size=d.get("batch_size", 256),
                               show_progress_bar=False, normalize_embeddings=True,
                               convert_to_numpy=True).astype(dt)
    mm.flush()
    del mm, texts
    gc.collect()
    with open(ids_p, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    return f"embedded {len(doc_ids):,} docs -> {dt.__name__}"


# =========================================================================== #
# SECTION 2: BM25 tuning
# =========================================================================== #
def _bm25_index(texts, conf, stemmer):
    import bm25s
    tok = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
    r = bm25s.BM25(method=conf.get("method", "lucene"), k1=conf["k1"], b=conf["b"])
    r.index(tok, show_progress=False)
    return r, tok


def sec_tune_bm25(cfg, paths):
    import bm25s
    try:
        import Stemmer
    except ImportError:
        Stemmer = None
    name = cfg["dataset"]
    out_j = os.path.join(paths["bm25_tuning"], f"{name}_bm25_best.json")
    if os.path.exists(out_j):
        with open(out_j, encoding="utf-8") as f:
            b = json.load(f)
        return f"already tuned: k1={b['k1']} b={b['b']} stem={b['use_stemming']}"
    if not cfg.get("bm25_tuning", {}).get("enabled", False):
        c = cfg["bm25"]
        return (f"SKIPPED (bm25_tuning.enabled=false) -- using config values "
                f"k1={c['k1']} b={c['b']} stem={c['use_stemming']}")

    folder = dataset_dir(paths, name)
    with open(os.path.join(processed_dir(paths, name, create=False),
                           "corpus_ids.json"), encoding="utf-8") as f:
        cid = json.load(f)
    texts = read_corpus_texts(folder, cid)
    queries, qrels = read_queries(folder), read_qrels(folder, cfg.get("split", "test"))
    qids = [q for q in queries if q in qrels and qrels[q]]
    k = cfg["retrieval"].get("eval_k", 10)
    g = cfg["bm25_tuning"]
    rows = []
    for stem in g["use_stemming"]:
        st = Stemmer.Stemmer("english") if (stem and Stemmer) else None
        ctok = bm25s.tokenize(texts, stopwords="en", stemmer=st, show_progress=False)
        qtok = bm25s.tokenize([queries[q] for q in qids], stopwords="en", stemmer=st,
                              show_progress=False)
        for k1 in g["k1"]:
            for b in g["b"]:
                r = bm25s.BM25(method=cfg["bm25"].get("method", "lucene"), k1=k1, b=b)
                r.index(ctok, show_progress=False)
                idx, _ = r.retrieve(qtok, k=k, show_progress=False)
                tot = n = 0
                for i, q in enumerate(qids):
                    rels = {d: int(v) for d, v in qrels[q].items() if int(v) > 0}
                    v = ndcg([cid[int(j)] for j in idx[i]], rels, k) if rels else None
                    if v is not None:
                        tot += v
                        n += 1
                rows.append(dict(k1=k1, b=b, use_stemming=bool(stem),
                                 score=tot / max(n, 1)))
                print(f"\r  k1={k1} b={b} stem={int(stem)} -> {rows[-1]['score']:.4f}",
                      end="", flush=True)
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(os.path.join(paths["bm25_tuning"], f"{name}_bm25_tuning.csv"), index=False)
    best = df.iloc[0].to_dict()
    with open(out_j, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return (f"\n  BEST k1={best['k1']} b={best['b']} stem={best['use_stemming']} "
            f"({best['score']:.4f}) -- copy into config.yaml `bm25`")


# =========================================================================== #
# SECTION 3: retrieval (cached per split, fusion-independent)
# =========================================================================== #
def dense_topk(q_emb, corpus_emb, top_k, dev, dtype, chunk, qbatch):
    import torch
    n_doc, dim = corpus_emb.shape
    k = min(top_k, n_doc)
    nq = q_emb.shape[0]
    oi = np.empty((nq, k), dtype=np.int64)
    ov = np.empty((nq, k), dtype=np.float32)
    bpe = 2 if dtype == torch.float16 else 4
    Cg = None
    if dev == "cuda" and n_doc * dim * bpe < 18e9:
        try:
            Cg = torch.from_numpy(np.array(corpus_emb[:])).to(dev).to(dtype)
        except torch.cuda.OutOfMemoryError:
            Cg = None
            torch.cuda.empty_cache()
    # The similarity block is ALWAYS chunked over the corpus: Q @ C.T over 5.2M
    # docs would be ~40 GB for a 2048-query batch. Preloading only avoids
    # re-reading the corpus per batch.
    for qs in tqdm(range(0, nq, qbatch), desc="  dense", leave=False):
        qe = min(qs + qbatch, nq)
        Q = torch.from_numpy(np.ascontiguousarray(q_emb[qs:qe])).to(dev).to(dtype)
        rv = ri = None
        for s in range(0, n_doc, chunk):
            e = min(s + chunk, n_doc)
            blk = Cg[s:e] if Cg is not None else \
                torch.from_numpy(np.array(corpus_emb[s:e])).to(dev).to(dtype)
            sims = Q @ blk.T
            kk = min(k, e - s)
            v, i = torch.topk(sims, kk, dim=1)
            i = i + s
            if rv is None:
                rv, ri = v, i
            else:
                rv = torch.cat([rv, v], 1)
                ri = torch.cat([ri, i], 1)
                v, sel = torch.topk(rv, min(k, rv.shape[1]), 1)
                rv, ri = v, torch.gather(ri, 1, sel)
            del sims
            if Cg is None:
                del blk
        oi[qs:qe], ov[qs:qe] = ri.cpu().numpy(), rv.float().cpu().numpy()
        del Q, rv, ri
    del Cg
    return oi, ov


def sec_retrieve(cfg, paths):
    import bm25s
    import torch
    from sentence_transformers import SentenceTransformer
    try:
        import Stemmer
    except ImportError:
        Stemmer = None

    name = cfg["dataset"]
    folder = dataset_dir(paths, name)
    pdir = processed_dir(paths, name, create=False)
    top_k = cfg["retrieval"]["top_k"]
    splits = cfg["pipeline"]["splits"]

    def cache_p(s):
        return os.path.join(pdir, f"retrieval_{s}_top{top_k}.npz")

    # A cache written before q_emb was added holds valid BM25/dense results but
    # no query embeddings. Re-encoding queries takes seconds, so REPAIR those
    # instead of re-running retrieval (which would cost ~15 min for dev+test).
    todo, repair = [], []
    for s in splits:
        if read_qrels(folder, s) is None:
            continue
        if not os.path.exists(cache_p(s)):
            todo.append(s)
        elif "q_emb" not in np.load(cache_p(s), allow_pickle=False).files:
            repair.append(s)
    if not todo and not repair:
        return f"already cached for {splits}"

    with open(os.path.join(pdir, "corpus_ids.json"), encoding="utf-8") as f:
        cid = json.load(f)
    d = cfg["dense"]
    dev = "cuda" if (d.get("device", "cuda") != "cpu" and torch.cuda.is_available()) else "cpu"
    dtype = (torch.float16 if (dev == "cuda" and
             str(d.get("embedding_dtype", "float32")).lower() == "float16") else torch.float32)
    model = SentenceTransformer(d["model_name"], device=dev)
    if d.get("max_seq_length"):
        model.max_seq_length = d["max_seq_length"]
    queries = read_queries(folder)

    for s in repair:                       # add q_emb only; keep the ranked lists
        z = np.load(cache_p(s), allow_pickle=False)
        qids = [str(q) for q in z["qids"]]
        print(f"  [{s}] repairing cache: encoding {len(qids):,} queries (lists kept)")
        qe = model.encode([queries[q] for q in qids], batch_size=d.get("batch_size", 256),
                          show_progress_bar=True, normalize_embeddings=True,
                          convert_to_numpy=True).astype(np.float32)
        np.savez(cache_p(s), qids=np.asarray(qids), bm_idx=z["bm_idx"], bm_val=z["bm_val"],
                 dn_idx=z["dn_idx"], dn_val=z["dn_val"], q_emb=qe)
    if not todo:
        return f"repaired {repair} (q_emb added; retrieval reused)"

    print(f"  loading corpus text ({len(cid):,} docs) + BM25 index")
    texts = read_corpus_texts(folder, cid)
    st = Stemmer.Stemmer("english") if (cfg["bm25"].get("use_stemming") and Stemmer) else None
    retr, _ = _bm25_index(texts, cfg["bm25"], st)
    del texts
    gc.collect()
    corpus_emb = np.load(os.path.join(pdir, "corpus_emb.npy"), mmap_mode="r")

    for s in todo:
        qr = read_qrels(folder, s)
        qids = [q for q in queries if q in qr and qr[q]]
        qt = [queries[q] for q in qids]
        print(f"  [{s}] {len(qids):,} queries: BM25 ...")
        tok = bm25s.tokenize(qt, stopwords="en", stemmer=st, show_progress=False)
        kk = min(top_k, len(cid))
        try:    # n_threads is a large win here; not present on older bm25s
            bi, bv = retr.retrieve(tok, k=kk, show_progress=True, n_threads=N_THREADS)
        except TypeError:
            bi, bv = retr.retrieve(tok, k=kk, show_progress=True)
        print(f"  [{s}] dense ...")
        qe = model.encode(qt, batch_size=d.get("batch_size", 256), show_progress_bar=True,
                          normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        di, dv = dense_topk(qe, corpus_emb, top_k, dev, dtype,
                            int(d.get("retrieval_chunk_size", 50000)),
                            int(d.get("query_batch_size", 2048)))
        np.savez(os.path.join(pdir, f"retrieval_{s}_top{top_k}.npz"),
                 qids=np.asarray(qids), bm_idx=bi, bm_val=bv.astype(np.float32),
                 dn_idx=di, dn_val=dv.astype(np.float32), q_emb=qe)
        print(f"  [{s}] cached.")
    return f"retrieved {todo}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--only", dest="only", type=int, default=None)
    ap.add_argument("--to", dest="end", type=int, default=9)
    args = ap.parse_args()

    cfg = load_config()
    paths = get_paths(cfg)
    import sections                      # sections 4-9 (imports core, not this)
    SECTIONS = [
        ("download", sec_download), ("embed", sec_embed), ("tune_bm25", sec_tune_bm25),
        ("retrieve", sec_retrieve), ("dataset", sections.sec_dataset),
        ("screen", sections.sec_screen), ("ablate", sections.sec_ablate),
        ("rescreen", sections.sec_rescreen), ("final_fit", sections.sec_final_fit),
        ("benchmark", sections.sec_benchmark),
    ]
    todo = ([args.only] if args.only is not None
            else list(range(args.start, min(args.end, len(SECTIONS) - 1) + 1)))

    print(f"### Query-Adaptive Score Fusion | dataset={cfg['dataset']} "
          f"| fusion={cfg['fusion']['function']}/{cfg['fusion']['normalizer']} "
          f"| metric=NDCG@{cfg['retrieval'].get('eval_k', 10)}")
    t0 = time.perf_counter()
    for i in todo:
        label, fn = SECTIONS[i]
        print(f"\n=== [{i}] {label} " + "=" * (60 - len(label)))
        t = time.perf_counter()
        msg = fn(cfg, paths)
        print(f"--- [{i}] {label}: {msg}  ({time.perf_counter()-t:.1f}s)")
    print(f"\n### done in {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
