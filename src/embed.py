"""embed.py -- embed the documents and queries of the active dataset.

Loads the BEIR dataset named in config.yaml, encodes every document
(title + text) and every query with all-mpnet-base-v2, and caches the
L2-normalised embeddings plus their id order to data/processed_data/<dataset>/:

    corpus_emb.npy   (Ndoc x 768)   corpus_ids.json  (row -> doc_id)
    query_emb.npy    (Nq   x 768)   query_ids.json   (row -> qid)

Scales to very large corpora (e.g. MS MARCO, 8.8M docs) without OOM:
  * the corpus dict is freed as soon as the texts are extracted;
  * documents are encoded SHARD BY SHARD and streamed straight into a .npy
    MEMMAP, so the full embedding matrix is never resident in RAM;
  * embeddings are stored in `dense.embedding_dtype` (default float32, full
    precision; set to float16 in config.yaml to halve RAM/disk if needed).

Progress: an outer tqdm bar over shards (inner per-batch bar suppressed to keep
logs readable on multi-million-doc corpora).
"""

import os
import sys
import gc
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader

from utils import load_config, get_paths, dataset_dir, processed_dir

DTYPES = {"float16": np.float16, "float32": np.float32}


def build_doc_text(doc):
    """BEIR docs are {'title': ..., 'text': ...}; concatenate when titled."""
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return (title + " " + text).strip() if title else text


def _resolve_device(requested):
    import torch
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    print("[embed] CUDA not available -- falling back to CPU.")
    return "cpu"


def main():
    config = load_config()
    paths = get_paths(config)
    name = config["dataset"]
    split = config.get("split", "test")
    d_conf = config["dense"]

    corpus, queries, _ = GenericDataLoader(data_folder=dataset_dir(paths, name)).load(split=split)
    out_dir = processed_dir(paths, name)

    device = _resolve_device(d_conf.get("device", "cuda"))
    print(f"[embed] loading model '{d_conf['model_name']}' on {device}")
    model = SentenceTransformer(d_conf["model_name"], device=device)
    if d_conf.get("max_seq_length"):
        model.max_seq_length = d_conf["max_seq_length"]

    dim = model.get_sentence_embedding_dimension()
    dtype = DTYPES.get(str(d_conf.get("embedding_dtype", "float32")).lower(), np.float32)
    normalize = d_conf.get("normalize", True)
    batch_size = d_conf.get("batch_size", 256)
    shard = int(d_conf.get("encode_shard_size", 100000))

    # ---- documents: extract texts, FREE the corpus dict, then shard -> memmap ----
    doc_ids = list(corpus.keys())
    doc_texts = [build_doc_text(corpus[d]) for d in doc_ids]
    del corpus
    gc.collect()

    n = len(doc_ids)
    emb_path = os.path.join(out_dir, "corpus_emb.npy")
    print(f"[embed] '{name}': encoding {n:,} docs -> memmap {dtype.__name__} "
          f"({n} x {dim}), shard={shard:,}")
    mm = open_memmap(emb_path, mode="w+", dtype=dtype, shape=(n, dim))
    for start in tqdm(range(0, n, shard), desc="[embed] corpus shards", unit="shard"):
        end = min(start + shard, n)
        vec = model.encode(
            doc_texts[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        mm[start:end] = vec.astype(dtype)
        del vec
    mm.flush()
    del mm, doc_texts
    gc.collect()

    with open(os.path.join(out_dir, "corpus_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)

    # ---- queries (small; one pass) ----
    qids = list(queries.keys())
    q_texts = [queries[q] for q in qids]
    print(f"[embed] '{name}': encoding {len(q_texts):,} queries")
    query_emb = model.encode(
        q_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    ).astype(dtype)
    np.save(os.path.join(out_dir, "query_emb.npy"), query_emb)
    with open(os.path.join(out_dir, "query_ids.json"), "w", encoding="utf-8") as f:
        json.dump(qids, f)

    print(f"[embed] saved -> {out_dir}\n"
          f"        corpus_emb ({n} x {dim}, {dtype.__name__}) | "
          f"query_emb {query_emb.shape}")


if __name__ == "__main__":
    main()
