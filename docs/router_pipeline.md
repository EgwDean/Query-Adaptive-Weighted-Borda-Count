# Router Training Pipeline (Phase 2)

How we go from the feature dataset ([feature_dataset.md](feature_dataset.md)) to
a trained router that predicts the fusion weight `alpha` per query.

**Stages** (each is a separate script, run in order):

| # | Stage | Script | Status |
|---|-------|--------|--------|
| 1 | Screen model families x framings | `src/screen_routers.py` | done |
| 2 | Feature ablation (best model from #1) | *(planned)* | — |
| 3 | Re-screen families + params on the ablated features | *(planned)* | — |
| 4 | Final fit on the full train split | *(planned)* | — |
| 5 | Benchmark vs all baselines + SHAP | *(planned)* | — |

---

## The three governing rules

**1. Select on NDCG@10, never on MSE/CE.**
The `alpha -> NDCG` curve is flat over wide plateaus, so a large alpha error can
cost zero NDCG and a small one can cost a lot. Worse, on hotpotqa an *MSE-optimal*
predictor is actively **bad**: the label mean is 0.42, and predicting 0.42 for
every query scores ~0.567 — **0.078 below** the trivial best-constant baseline
(0.645). MSE/CE are logged as diagnostics only.

**2. Split discipline.** Fit on **train**, select on **dev** (family, params,
features, everything). **Test is opened exactly once**, at stage 5.

**3. Scalers are fit inside the pipeline**, so they only ever see train.

---

## Stage 1 — `screen_routers.py`

For every `(family, framing)` pair, run an **independent Optuna/TPE study** over
that family's hyperparameters. Independent studies (rather than one BO over a
mixed, conditional space) give every family the same trial budget, so the
comparison is best-vs-best and fair.

**Families** (`config.yaml: router_screen.families`) — SVM and KNN are
deliberately excluded: SVM is O(n^2)+ and dies at the 80k final fit, and KNN pays
O(n_train) *at inference*, which defeats the point of a ~1 ms router.

`lightgbm`, `xgboost`, `catboost`, `hist_gbdt`, `random_forest`, `extra_trees`,
`elasticnet` (regression only), `logreg` (classification only), `mlp`.

**Framings** (`router_screen.framings`) — how alpha is turned into a target:

| Framing | Target | Predicted alpha |
|---|---|---|
| `regression` | alpha itself | `clip(predict(X), 0, 1)` |
| `binary` | `alpha > 0.5` | `P(class = 1)` |
| `multibin` | nearest of `n_bins` centres | `sum(proba * bin_centres)` |

**Searched alongside the model hyperparameters:**
- `scaler`: `standard` / `robust` / `quantile` (scale-sensitive families only;
  trees get none). Not fixed in advance because features are heavy-tailed —
  `nqc_bm25` spans 0.5 to 30, and StandardScaler does not fix skew.
- `use_sample_weight`: on/off (see below). Forced off for `mlp` (sklearn's MLP
  does not accept `sample_weight`).

### Sample weighting by `alpha_sensitivity`

`alpha_sensitivity = max(curve) - min(curve)` = how much alpha matters for that
query. On hotpotqa **20.3% of queries are flat** (sensitivity 0) — and *all* of
them were labelled `alpha = 0.0` purely by the lowest-wins tie-break, making up
**72% of the entire alpha=0 spike**. Their labels are fabricated. Used as a
sample weight, those rows contribute ~0 to the loss instead of teaching the model
to predict 0. Searched on/off rather than assumed.

### Degeneracy check

The best constant alpha already scores ~0.645, so a model predicting ~0.99 for
every query scores well **while having learned nothing**. Each config therefore
reports:
- `pred_alpha_mean`, `pred_alpha_std` — spread of the predictions
- `pred_oracle_corr` — Pearson corr. with the oracle alpha
- `degenerate` — flagged when `pred_alpha_std < 0.01`

A high NDCG with `degenerate = True` is the constant baseline wearing a hat.

### Significance

Per-query dev NDCG is bootstrapped (`bootstrap_resamples`, default 1000):
- `ci_lo`/`ci_hi` — 95% CI of each config's mean
- `diff_vs_best` + `diff_ci_lo`/`diff_ci_hi` — **paired** bootstrap of the
  difference against the top config; `significant` = the CI excludes 0.

Paired (not overlapping CIs) because it is far more powerful: two configs can
have heavily overlapping CIs while one still beats the other on nearly every query.

### Reference rows

`constant_alpha` (tuned on **train** only) and `oracle` are included so the model
scores are interpretable. These are **not** the formal benchmark — that is
stage 5, against the full baseline set in [comparison_methods.md](comparison_methods.md).

### Outputs (`data/results/router_screening/`)

| File | Contents |
|---|---|
| `<ds>_router_screening.csv` | every trial (`is_best` marks each study's winner) |
| `<ds>_router_screening_best_per_config.csv` | one row per config: NDCG, CIs, paired diff, degeneracy |
| `<ds>_router_screening_best.json` | the winning family/framing/params + feature list |

### Config

```yaml
router_screen:
  train_subset: 10000        # seeded sample of train used for screening
  n_trials: 50               # Optuna trials PER (family, framing)
  bootstrap_resamples: 1000
  n_bins: 11                 # for the multibin framing
  families: [...]
  framings: [regression, binary, multibin]
```

---

## What stage 1 must beat

From the hotpotqa test curve (dev will be similar):

```
pure dense  (alpha=0.00)   0.3929
pure BM25   (alpha=1.00)   0.6300
best static (alpha=0.99)   0.6451   <- the bar
oracle      (per-query)    0.7364   <- the ceiling
headroom                   0.0913   (+14.2% relative)
```

BM25 dominates dense overall, yet **25% of queries still want alpha <= 0.1**.
Dense-preferring queries have shallow curves (little lost by forcing BM25);
BM25-preferring queries have steep ones (much lost by forcing dense). Averaging
therefore pushes the global optimum to alpha≈0.99 even though the *median*
per-query optimum is 0.23. That asymmetry is exactly the gap only per-query
routing can close.
