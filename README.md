# Query-Adaptive Weighted Borda Count

A hybrid retrieval system that fuses a **lexical** and a **semantic** retriever
with **Weighted Borda Count**, where the fusion weight `alpha` is learned
**per query** instead of being fixed for the whole collection. Part of ongoing
research on hybrid retrieval.

```
score(d) = alpha * (N - rank_sparse(d)) + (1 - alpha) * (N - rank_dense(d))
```

* `alpha = 1` → pure **BM25** (lexical)
* `alpha = 0` → pure **dense** (semantic)
* `N` = Borda list length (top-`k` retrieval depth); a document missing from a
  list scores 0 from that list.

This repository currently implements **Phase 1: dataset selection** — finding
the BEIR collection that best balances lexical and semantic signal, so it is the
strongest test bed for the adaptive system later on.

---

## Why parity matters

Per-query routing can only help when the two retrievers genuinely *disagree*
per query. If one retriever dominates, the per-query optimal `alpha` collapses
to a corner (all 0 or all 1) and adaptive fusion has nothing to gain.

Two deliberate choices keep the retrievers in the same strength class:

| Role     | Model                                   | Notes |
|----------|-----------------------------------------|-------|
| Lexical  | **BM25** (Okapi, tuned `k1`/`b`)        | `bm25s`, fast in-RAM sparse index |
| Semantic | **all-mpnet-base-v2** (768-d, cosine)   | chosen for *strength parity* with tuned BM25, not raw power |

A stronger dense model (e5-large, bge-large, BGE-M3) would dominate BM25 and
flatten the `alpha` distribution — the opposite of what we need here.

---

## What Phase 1 measures

For each query we compute the **oracle alpha**: the `alpha` (grid-searched over
`[0, 1]` in steps of 0.01) that maximises **NDCG@100** of the Weighted Borda
fusion (lowest alpha wins ties). The spread of oracle alphas across a dataset's
queries tells us how balanced it is:

* **Wide spread (high IQR)** → queries need different blends → great test bed.
* **Median near 0.5** → lexical and semantic are, on average, equally useful.
* **Collapsed near 0 or 1** → one retriever dominates → weak test bed.

The dataset with the highest spread (or median closest to 0.5) is the one to
carry forward.

---

## Repository layout

```
.
├── config.yaml              # all settings + BEIR dataset catalogue
├── requirements.txt         # pinned dependencies
├── commands.txt             # Linux/HPC setup + run commands
├── clean.sh                 # delete datasets/embeddings, keep results
├── README.md
├── docs/
│   ├── ltr_router_features.md  # Phase-2 router feature catalogue (QPP + routing)
│   └── comparison_methods.md   # baselines & methods to compare against
├── src/
│   ├── utils.py             # config + path helpers
│   ├── download.py          # download one BEIR dataset (tqdm)
│   ├── embed.py             # embed docs + queries (all-mpnet-base-v2, tqdm)
│   ├── alpha_distribution.py# BM25 + dense + oracle alpha + boxplots
│   └── pipeline.py          # run download -> embed -> alpha_distribution
└── data/
    ├── datasets/<name>/        # raw BEIR corpus, queries, qrels
    ├── processed_data/<name>/  # cached embeddings + id maps
    └── results/                # *_alpha.csv, boxplots, alpha_summary.csv
```

---

## Setup (Linux / HPC)

See [commands.txt](commands.txt) for copy-paste commands. In short:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121  # match your CUDA
pip install -r requirements.txt
```

Install **torch first**, matched to the cluster CUDA, before the rest.

---

## Running (once per dataset)

Pick a dataset by editing `dataset:` in [config.yaml](config.yaml) (the file
lists every BEIR dataset with corpus size, query count and download size), then:

```bash
python src/download.py             # 1. download the BEIR dataset
python src/embed.py                # 2. embed docs + queries
python src/alpha_distribution.py   # 3. retrieve, oracle alpha, boxplots
```

Each step is cache-friendly and reports progress with `tqdm`. Repeat for every
dataset you want to compare. Start with the small ones (`scifact`, `nfcorpus`,
`fiqa`, `scidocs`, `arguana`) plus the lexical-leaning `trec-covid` and
`webis-touche2020`; leave the million-doc sets for last.

### Outputs (in `data/results/`)

| File | Description |
|------|-------------|
| `<name>_alpha.csv` | per-query: `alpha`, `oracle_ndcg`, `bm25_ndcg`, `dense_ndcg`, `n_rel` |
| `<name>_alpha_boxplot.png` | oracle alpha boxplot for that dataset |
| `combined_alpha_boxplot.png` | all processed datasets side by side |
| `alpha_summary.csv` | per-dataset mean/median/std/IQR of alpha, mean NDCGs, ranked by spread |

`alpha_distribution.py` also prints which dataset has the **highest alpha
spread** and which has the **median closest to 0.5**.

---

## Configuration highlights ([config.yaml](config.yaml))

* `dataset` / `split` — the active dataset and its qrels split.
* `bm25` — `method`, `k1`, `b`, `use_stemming`.
* `dense` — model name, `batch_size`, `max_seq_length`, `device`.
* `retrieval.top_k` — retrieval depth (the two ranked lists).
* `borda` — `N` and the `alpha` grid (`alpha_min`/`alpha_max`/`alpha_step`).

---

## Method notes

* **NDCG@100** uses gain `2^rel - 1` (trec_eval / BEIR convention), so graded
  qrels (e.g. `trec-covid`, `nfcorpus`) are handled correctly.
* Queries with no relevant documents are excluded from the alpha computation.
* BM25 runs on `bm25s` (vectorised sparse, in-RAM); dense retrieval is exact
  cosine (`top_k` via `torch.topk`) over the cached normalised embeddings.
* Everything is deterministic given the config (fixed BM25 params, fixed alpha
  grid, stable tie-breaking).

---

## Documentation

* [docs/ltr_router_features.md](docs/ltr_router_features.md) — Phase-2 router
  feature catalogue: query performance predictors (pre-/post-retrieval, dense)
  plus hybrid routing signals, each with a plain-math description and a
  **"Lex/sem routing?"** column marking whether the source used it directly to
  choose between lexical and dense retrieval (**Yes**) or is a general QPP signal
  repurposed here as a per-retriever router input (**No**).
* [docs/comparison_methods.md](docs/comparison_methods.md) — the baselines and
  retrieval paradigms to evaluate against (single retrievers, static fusion,
  adaptive fusion, SPLADE, reranker, oracle ceiling).

---

## Roadmap

* **Phase 1 (this repo):** choose the most lexical+semantic-balanced dataset.
* **Phase 2:** learning-to-rank features
  ([docs/ltr_router_features.md](docs/ltr_router_features.md)) + a router that
  predicts `alpha` per query; compare Weighted Borda Count against a
  **globally-tuned static** weight and against the best single retriever
  ([docs/comparison_methods.md](docs/comparison_methods.md)).
