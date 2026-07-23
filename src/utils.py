"""Shared helpers: configuration loading and path management.

Every script imports from here so paths and config are resolved consistently
regardless of the current working directory.
"""

import os
import yaml


def repo_root():
    """Absolute path to the repository root (one level above src/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))


def load_config(path=None):
    """Load config.yaml from the repo root (or an explicit path)."""
    if path is None:
        path = os.path.join(repo_root(), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths(config, create=True):
    """Resolve the data directories, optionally creating them."""
    root = repo_root()
    p = config["paths"]
    data_dir = os.path.join(root, p["data_dir"])
    results = os.path.join(data_dir, p["results_subdir"])
    paths = {
        "root": root,
        "data": data_dir,
        "datasets": os.path.join(data_dir, p["datasets_subdir"]),
        "processed": os.path.join(data_dir, p["processed_subdir"]),
        "results": results,
        # per-phase sub-folders inside results/
        "alpha_results": os.path.join(results, p.get("alpha_results_subdir", "alpha_distribution")),
        "bm25_tuning": os.path.join(results, p.get("bm25_tuning_subdir", "bm25_tuning")),
        "feature_dataset": os.path.join(results, p.get("feature_dataset_subdir", "feature_dataset")),
        "router_screening": os.path.join(results, p.get("router_screening_subdir", "router_screening")),
        "router_final": os.path.join(results, p.get("router_final_subdir", "router_final")),
    }
    if create:
        for key in ("datasets", "processed", "results", "alpha_results", "bm25_tuning",
                    "feature_dataset", "router_screening", "router_final"):
            os.makedirs(paths[key], exist_ok=True)
    return paths


def dataset_dir(paths, name):
    """data/datasets/<name>/ -- raw BEIR corpus, queries, qrels."""
    return os.path.join(paths["datasets"], name)


def build_doc_text(doc):
    """BEIR docs are {'title': ..., 'text': ...}; concatenate when titled. Shared
    across sections so document text is built identically everywhere."""
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return (title + " " + text).strip() if title else text


def processed_dir(paths, name, create=True):
    """data/processed_data/<name>/ -- cached embeddings + id maps."""
    d = os.path.join(paths["processed"], name)
    if create:
        os.makedirs(d, exist_ok=True)
    return d
