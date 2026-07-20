# Query-Adaptive Score Fusion

Hybrid retrieval that fuses a **lexical** and a **semantic** retriever with a
**convex combination of normalised scores**, where the fusion weight `alpha` is
predicted **per query** by a cheap router instead of being fixed globally.

```
fuse(d) = alpha * norm(bm25_score(d)) + (1 - alpha) * norm(dense_score(d))
```

* `alpha = 1` → pure **BM25** (lexical) · `alpha = 0` → pure **dense** (semantic)
* `norm` = per-query min-max over each retriever's own top-k

**Claim under test:** a per-query `alpha` beats the best single **global**
`alpha`.

**Why score fusion and not RRF/Borda?** Rank fusion keeps only *position* and
throws away score *magnitude* — the information that says "doc A and B are both
excellent, C is junk", not merely "A > B > C". This is a settled point in the
literature (Bruch, Gai & Ingber, *An Analysis of Fusion Functions for Hybrid
Retrieval*, ACM TOIS 2023), **not** a contribution of this work. RRF and Borda
appear only as standard baselines.

---

## Run it

Everything is one command. Each section skips if its outputs already exist.

```bash
python src/pipeline.py            # run all sections
python src/pipeline.py --from 4   # force re-run from section 4 onward
python src/pipeline.py --only 9   # a single section
```

| # | Section | Writes |
|---|---------|--------|
| 0 | `download` | `data/datasets/<ds>/` |
| 1 | `embed` | `corpus_emb.npy` (memmap, sharded) |
| 2 | `tune_bm25` | `results/bm25_tuning/` *(off by default; values already tuned)* |
| 3 | `retrieve` | `retrieval_{split}_top1000.npz` — ranked lists **+ raw scores** |
| 4 | `dataset` | router features + **alpha→NDCG curve** + oracle `alpha` label |
| 5 | `screen` | model families × framings (Optuna, selected on dev) |
| 6 | `ablate` | greedy backward feature elimination |
| 7 | `rescreen` | families × framings × feature-set sizes |
| 8 | `final_fit` | refit on the full train split → **frozen** `router.joblib` |
| 9 | `benchmark` | all baselines vs the router on **test — opened once** |

**Section 3 is cached and fusion-independent.** Changing `fusion.function` only
re-runs section 4 onward (minutes), never retrieval (hours).

---

## Layout

```
config.yaml            # every setting, documented inline
src/
  pipeline.py          # ENTRY POINT: orchestration + sections 0-3
  core.py              # BEIR I/O, metrics, fusion functions, alpha curve, bootstrap
  sections.py          # sections 4-9 (dataset -> router -> benchmark)
  utils.py             # config + path helpers
data/
  datasets/<ds>/       # raw BEIR
  processed_data/<ds>/ # embeddings + cached retrieval
  results/             # bm25_tuning, feature_dataset, router_screening, router_final
```

---

## Method notes

* **Metric:** NDCG@10 (`retrieval.eval_k`). Candidate pool `top_k=1000`
  (BEIR practice: retrieve deep, evaluate shallow).
* **Split discipline:** fit on **train**, select on **dev** (family,
  hyperparameters, features — everything), open **test exactly once** in
  section 9.
* **The alpha→NDCG curve** stores NDCG at all 101 alphas for every query. Any
  predicted alpha can then be scored by table lookup instead of re-running
  retrieval — and it yields `alpha_sensitivity` (max−min), used as a sample
  weight so queries whose curve is flat don't train the model on labels that are
  pure tie-break artifacts.
* **Decision rule = histogram binning.** A model's raw output is a proxy or a
  probability, *not* a fusion weight; used directly it loses to the constant
  baseline. Instead the output only **ranks** queries into bins, and each bin
  emits the alpha maximising its average NDCG curve (learned on a held-out slice
  of train). With no signal every bin picks the same alpha → exactly the
  constant baseline, so this rule **cannot do worse**.
* **Selection under ties:** results cluster tightly, so among configs
  statistically tied with the best (paired bootstrap) we take the **cheapest** —
  fewest features, then lowest inference-cost class.
* **Threads:** `n_jobs=-1` (=32) oversubscribes and **hangs** on a shared box.
  Capped to 8 via `PIPE_THREADS`; never set `-1`.

---

## Documentation

* [docs/comparison_methods.md](docs/comparison_methods.md) — the baselines
* [docs/ltr_router_features.md](docs/ltr_router_features.md) — QPP/routing feature catalogue
* [docs/inference_feature_inventory.md](docs/inference_feature_inventory.md) — which features are inference-time computable
* [docs/bm25_parameter_history.md](docs/bm25_parameter_history.md) — BM25 params + metric changes per run

---

## Status

Pipeline rebuilt around convex score fusion; sections 0–3 reuse the existing
cached corpus embeddings and BM25 tuning. **No results yet under this fusion** —
sections 4–9 need a run.
