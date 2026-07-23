# Query-Adaptive Score Fusion

A study of query-adaptive fusion in hybrid retrieval: **when a per-query fusion
weight helps, why it usually does not, and how to deploy it safely.**

Two retrievers — BM25 (lexical) and `all-mpnet-base-v2` (dense) — are combined by
a convex combination of per-query min-max normalised scores:

```
fuse(d) = α · norm(bm25_score(d))  +  (1 − α) · norm(dense_score(d))
```

`α = 1` is pure lexical, `α = 0` pure semantic. A cheap **router** predicts `α`
per query. The question is whether that per-query `α` beats the best single
global `α`.

Score fusion is primary; RRF and Borda appear only as baselines (that score
fusion keeps magnitude and rank fusion discards it is settled — Bruch, Gai &
Ingber, *An Analysis of Fusion Functions for Hybrid Retrieval*, TOIS 2023).

## Findings

The study covers **6 BEIR datasets × 3 fusion functions**. `hotpotqa` is the
development dataset; the other five are held out and inherit its frozen router
spec (only weights and the calibration table are refit).

- **H2 — a router's raw output is not a fusion weight.** Across 18 model
  configurations, *no* raw-output router beats the best constant `α` (best
  0.669 vs constant 0.691); *every* calibrated router does (18/18). Histogram-
  binning calibration re-maps the model's ranking onto the `α` axis and, with no
  signal, degrades exactly to the constant — so it cannot do worse. This is the
  main methodological result.
- **H1 — gain grows with retriever complementarity.** Across held-out cells,
  `corr(oracle-α spread, gain) = +0.60`. Complementarity is necessary but not
  sufficient: high spread makes gain available, but realising it also needs
  data (fever, 6.6k queries: +0.053 significant; nfcorpus, 323 queries: +0.005
  n.s.). Zero spread reliably means no gain — quora is significantly negative.
- **H3 — the pattern is fusion-invariant.** The same shape holds under score,
  RRF, and Borda.

Per-dataset results under the primary score fusion (NDCG@10 on test):

| dataset            | constant α | router | oracle | gain     | sig. |
|--------------------|-----------:|-------:|-------:|---------:|:----:|
| hotpotqa (dev)     | 0.6747     | 0.6810 | 0.7238 | +0.0063  | yes  |
| fever              | 0.7285     | 0.7375 | 0.8057 | +0.0089  | yes  |
| nfcorpus           | 0.3743     | 0.3727 | 0.4211 | −0.0016  | no   |
| scifact            | 0.7324     | 0.7289 | 0.7885 | −0.0035  | no   |
| fiqa               | 0.5128     | 0.5146 | 0.5787 | +0.0017  | no   |
| quora              | 0.9013     | 0.8986 | 0.9346 | −0.0027  | yes  |

`gain` is router − constant α; `sig.` is a paired bootstrap of the per-query
NDCG@10 difference **on test** (95% CI excluding zero). quora's significant
*negative* gain is the negative control: with zero oracle-α spread there is
nothing to route on, and 10k queries are enough to detect the small harm.

## Run it

The whole study is one command. Every section skips when its outputs already
exist, so a run is resumable.

```bash
# 1. environment (Linux / CUDA box; Ubuntu 24.04 needs a venv)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121   # CPU: drop the index-url
pip install -r requirements.txt

# 2. run the full (dataset × fusion) matrix, unattended
python src/run_study.py                 # resumable; --status / --dry-run / --aggregate-only

# 3. figures + supporting experiments
python src/plot_alpha.py --split test   # oracle-α boxplots
python src/h2_decision_rule.py          # the raw-vs-calibrated table (H2)
python src/make_slides.py               # slides/query_adaptive_fusion.pptx
```

Long jobs on a shared box: run inside `tmux` (survives disconnects) and keep
`router.n_jobs` at 8 — `-1` oversubscribes a many-core machine and stalls.

### Single-dataset pipeline

`run_study.py` drives the pipeline per cell; you can also run it directly:

```bash
python src/pipeline.py            # all sections for config.yaml's dataset/fusion
python src/pipeline.py --from 4   # re-run from section 4 onward
python src/pipeline.py --only 9   # a single section
```

| # | Section | Writes |
|---|---------|--------|
| 0 | download   | `data/datasets/<ds>/` |
| 1 | embed      | `corpus_emb.npy` (memmap, sharded) |
| 2 | tune_bm25  | `results/bm25_tuning/` *(off by default; values pre-tuned)* |
| 3 | retrieve   | `retrieval_{split}_top1000.npz` — ranked lists **+ raw scores** |
| 4 | dataset    | router features + **α→NDCG curve** + oracle `α` label |
| 5 | screen     | model families × framings (Optuna, on dev) |
| 6 | ablate     | greedy backward feature elimination |
| 7 | rescreen   | families × framings × feature-set sizes |
| 8 | final_fit  | refit on the full train split → **frozen** `router.joblib` |
| 9 | benchmark  | all baselines vs the router on **test — opened once** |

Section 3 is cached and fusion-independent, so changing the fusion re-runs only
section 4 onward, not retrieval.

## Layout

```
config.yaml            every setting, documented inline
src/
  pipeline.py          entry point: orchestration + sections 0-3
  sections.py          sections 4-9 (dataset → router → benchmark)
  core.py              BEIR I/O, metrics, fusion functions, α curve, bootstrap
  utils.py             config + path helpers
  run_study.py         the full (dataset × fusion) study runner
  h2_decision_rule.py  the H2 experiment (raw vs calibrated)
  plot_alpha.py        oracle-α boxplots
  make_slides.py       builds the presentation deck
data/
  datasets/<ds>/       raw BEIR
  processed_data/<ds>/ embeddings + cached retrieval
  results/             bm25_tuning, feature_dataset, router_screening, router_final,
                       alpha_distribution
slides/                generated .pptx + figures
```

## Method notes

- **Metric:** NDCG@10 (`retrieval.eval_k`); candidate pool `top_k = 1000`.
- **Splits:** fit on train, select on dev (family, hyperparameters, features),
  open test exactly once (section 9). Held-out datasets select nothing.
- **Oracle α curve:** NDCG at all 101 alphas per query, stored so any predicted
  α is scored by table lookup; its max−min gives the sample weight that
  down-weights queries whose curve is flat.
- **Decision rule = histogram binning:** the model output only ranks queries into
  bins; each bin emits the α maximising its average NDCG curve. A
  `MIN_QUERIES_PER_BIN` floor prevents starved bins on small datasets.
- **Significance:** paired bootstrap of the per-query NDCG@10 difference.

## Documentation

- [docs/comparison_methods.md](docs/comparison_methods.md) — baselines
- [docs/ltr_router_features.md](docs/ltr_router_features.md) — router feature catalogue
- [docs/inference_feature_inventory.md](docs/inference_feature_inventory.md) — inference-time feature computation
- [docs/bm25_parameter_history.md](docs/bm25_parameter_history.md) — BM25 parameter / metric history
