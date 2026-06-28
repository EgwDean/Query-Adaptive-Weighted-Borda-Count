"""embed.py -- embed the documents and queries of the active dataset.

Loads the BEIR dataset named in config.yaml, encodes every document
(title + text) and every query with all-mpnet-base-v2, and caches the
L2-normalised embeddings plus their id order to data/processed_data/<dataset>/:

    corpus_emb.npy   (Ndoc x 768, float32)   corpus_ids.json  (row -> doc_id)
    query_emb.npy    (Nq   x 768, float32)   query_ids.json   (row -> qid)

Progress: sentence-transformers shows a tqdm bar per encode call.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader

from utils import load_config, get_paths, dataset_dir, processed_dir


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

    normalize = d_conf.get("normalize", True)
    batch_size = d_conf.get("batch_size", 256)

    # ---- documents ----
    doc_ids = list(corpus.keys())
    doc_texts = [build_doc_text(corpus[d]) for d in doc_ids]
    print(f"[embed] '{name}': encoding {len(doc_texts):,} documents")
    corpus_emb = model.encode(
        doc_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(os.path.join(out_dir, "corpus_emb.npy"), corpus_emb)
    with open(os.path.join(out_dir, "corpus_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)

    # ---- queries ----
    qids = list(queries.keys())
    q_texts = [queries[q] for q in qids]
    print(f"[embed] '{name}': encoding {len(q_texts):,} queries")
    query_emb = model.encode(
        q_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(os.path.join(out_dir, "query_emb.npy"), query_emb)
    with open(os.path.join(out_dir, "query_ids.json"), "w", encoding="utf-8") as f:
        json.dump(qids, f)

    print(
        f"[embed] saved -> {out_dir}\n"
        f"        corpus_emb {corpus_emb.shape} | query_emb {query_emb.shape}"
    )


if __name__ == "__main__":
    main()
