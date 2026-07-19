# Router Training Pipeline (Phase 2)

How we go from the feature dataset ([feature_dataset.md](feature_dataset.md)) to
a trained router that predicts the fusion weight `alpha` per query.

**Stages** (each is a separate script, run in order):

| # | Stage | Script | Status |
|---|-------|--------|--------|
| 1 | Screen model families x framings | `src/screen_routers.py` | done |
| 2 | Feature ablation (greedy backward) | `src/ablate_features.py` | ready |
| 3 | Re-screen families x framings x feature sets | `src/rescreen_routers.py` | ready |
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

### Decision rule: `raw` vs `calibrated` (searched)

**The model's output is not an alpha.** For the `binary` framing it is
`P(alpha > 0.5)` — a probability; for `regression` it is an MSE-fit estimate that
shrinks toward the label mean. Feeding either straight in as the fusion weight is
what makes every config lose to the constant baseline (measured: best raw model
0.6346 vs constant 0.6637 on hotpotqa dev).

- **`raw`** — use the output directly as alpha (the original behaviour).
- **`calibrated`** — **histogram binning**. The output is only a *score* used to
  rank/bin queries; the alpha comes from a lookup table:

```
split train_subset          -> 80% fit / 20% calibrate   (calib_fraction)
fit the model on the 80%
score the held-out 20%      -> honest scores (never the fitted rows: their
                               scores are overfit and would not transfer)
sort scores, cut into Q quantile bins   (n_calib_bins, searched: 10/20/50)
for each bin:
    bin_alpha = grid[ curve[rows_in_bin].mean(axis=0).argmax() ]
inference: score -> bin -> emit that bin's alpha
```

`curve[rows_in_bin].mean(axis=0)` averages the bin's queries down each of the
101 alpha columns, giving 101 averages; `argmax` keeps the best. **The bin index
constrains nothing** — every bin searches the full alpha grid, so bin 1 may well
emit 0.99.

**Floor = the constant baseline.** With no signal every bin's average curve is
the global curve, so every bin stores the same alpha and the rule reproduces the
constant exactly. Verified on synthetic curves: no-signal → 0.7094 vs constant
0.7094 (identical); with-signal → 0.7919 against a 0.8302 oracle.

Naming: the mechanism is **histogram binning** (Zadrozny & Elkan, ICML 2001;
their isotonic variant, KDD 2002 — verify before citing). The *problem* is
**cost-sensitive classification / policy learning with full feedback**: because
the curve gives the NDCG of every alpha for every query, the reward of every
action is known (this is full feedback, not bandit feedback). Isotonic regression
is the natural upgrade — monotonic, no bin count to choose.

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
  decision_rules: [raw, calibrated]
  calib_fraction: 0.2        # held-out share of train_subset for the calibration table
  n_calib_bins: [10, 20, 50] # quantile bins (searched; calibrated rule only)
  families: [...]
  framings: [regression, binary, multibin]
```

### Query budget per stage

| stage | queries | split |
|---|---|---|
| 1 (screen) | 10,000 | 8,000 fit / 2,000 calibrate |
| 3 (re-screen, post-ablation) | 10,000 | 8,000 fit / 2,000 calibrate |
| 4 (final fit) | ~85,000 | ~68,000 fit / ~17,000 calibrate |

The calibration table is far better estimated at stage 4 (~850 queries per bin
instead of ~100), which is exactly where it matters.

### Stage-1 RESULTS (hotpotqa, dev)

**Run A — `raw` rule (all 9 families x 3 framings).** Every one of the 25
configs **lost** to the constant (best `hist_gbdt|binary` 0.6346 vs 0.6637).
The decisive diagnostic: **the model with the best oracle correlation had the
worst NDCG** (`mlp|multibin` corr 0.397 → 0.6016; `hist_gbdt|binary` corr 0.292
→ 0.6346). The signal was real; the *decision rule* was broken. Hence
calibration.

**Run B — `calibrated` rule (7 fast families).** Every one of the 18 configs
**beat** the constant:

```
oracle                          0.7502
logreg|multibin   calib[20]     0.6767   <- WINNER
elasticnet|regression calib[10] 0.6759
lightgbm|regression   calib[10] 0.6758
...            (top 11 tied, significant=False)
random_forest|binary  calib[10] 0.6729   <- worst config, still > constant
constant alpha=0.99             0.6637   diff -0.0130 CI [-0.0159,-0.0099] SIG
```

**+0.0130 over the globally-tuned static alpha, statistically significant**
= 15% of the available oracle headroom (0.0865).

Findings:
1. **The decision rule, not the model, was the bottleneck.** Same features,
   same families: raw 0.60-0.635, calibrated 0.673-0.677.
2. **Calibration is an equalizer.** All 18 configs span just 0.0038; the top 11
   are statistically tied. The calibration layer does the work — the model only
   has to *rank* queries.
3. **Linear models won.** `logreg` and `elasticnet` top the table, so the
   nominal winner is also the cheapest to serve (logistic regression + a 20-bin
   lookup = microseconds), which underwrites the "~1 ms router" efficiency claim.
4. **No constant-reshuffling confound.** The calibrated routers emit
   `mean alpha ~= 0.91` while the baseline constant is 0.99, so we verified the
   best *possible* constant on dev: also **0.99 -> 0.6637**, identical to the
   train-tuned one. The gain therefore comes entirely from per-query variation
   (`pred_alpha_std ~= 0.18`), not from finding a better global alpha.
5. **`pred_oracle_corr` becomes misleading after calibration** (it fell 0.29 ->
   0.20 while NDCG rose): the output is a binned *action*, not a prediction of
   the oracle alpha. Keep it only as a degeneracy tripwire.

**Carried into stage 2:** `logreg | multibin | calib[20]` — nominally best,
statistically tied with 10 others, and by far the cheapest. Being linear, it
also makes the ablation meaningful: unlike trees, a linear model genuinely
suffers from junk features.

---

## Stage 2 — `ablate_features.py` (greedy backward elimination)

Start from all 46 features and prune one per round: try removing each survivor,
permanently drop the one whose removal **hurts least** by dev NDCG@eval_k.
**1,076 fits** (46+45+…+4, plus the full-set baseline) — never 2^46 subsets.

### Why greedy backward, not the alternatives

The feature set has known-redundant families (`avgIDF`/`SCS`/`AvICTF`/`γ1` all
measure specificity; `NQC`/`WIG`/`SMV`/`entropy` all measure dispersion — see
[ltr_router_features.md](ltr_router_features.md)). That redundancy breaks both
alternatives:

| Method | Behaviour on redundant features |
|---|---|
| Group cutting | drops a whole family, taking its one good member with it ❌ |
| Permutation importance | near-duplicates mask each other → **both** look useless → both dropped ❌ |
| **Greedy backward** | drops one twin; the survivor's contribution rises, so it is kept ✓ |

### Workhorse model

Defaults to the stage-1 winner, but that winner (`logreg`, `saga` solver) is
~14 s/fit → **~4 h** for the full path. `config.yaml: ablation` overrides it with
**`elasticnet|regression`**, which scored 0.6759 vs the winner's 0.6767 —
**statistically tied** (paired CI includes 0) — and is sub-second → **~5–15 min**.
Nothing is lost: **stage 3 re-screens every family** on the surviving features.

### Rules

- **Drop rule** — paired bootstrap of `(candidate − full set)` over the same dev
  queries. Paired, not independent CIs: two configs can have overlapping
  independent CIs while one wins query-by-query.
- **Final pick — parsimony, not argmax.** The **smallest** feature set whose
  paired CI vs the best path point includes 0. Taking the raw maximum overfits
  dev; parsimony is the defensible choice *and* yields a cheaper router.
- **Cost-aware tie-break.** Each feature is tagged by inference cost
  (`lookup` 14, `scores` 24, `invindex` 2, `embed` 4, `text` 2). When two
  removals tie, the **costlier** feature is dropped — making this a
  quality/latency Pareto result rather than pure accuracy chasing. The expensive
  ones are `clarity_*` (needs top-50 doc **text**) and
  `autocorr_*`/`apair_ratio_*` (embedding gathers + pairwise similarity).

### Caveat (must survive into the paper)

The greedy path takes ~1,000 maxima against dev, so the selected subset looks
**better on dev than it truly is** (selection bias). Contained by stage 3
re-screening and by test staying sealed until stage 5 — but treat the ablation
gain as **optimistic** until stage 5 confirms it.

### Outputs (`data/results/router_screening/`)

| File | Contents |
|---|---|
| `<ds>_ablation_path.csv` | one row per round — the **NDCG vs #features curve** (paper figure) |
| `<ds>_ablation_rounds.csv` | every candidate fit — the drop order = redundancy-aware importance ranking |
| `<ds>_ablation_best.json` | the chosen feature set, cost mix, parsimony rationale |

### Stage-2 RESULTS (hotpotqa, dev)

```
full 46 features    0.6755
best 11 features    0.6773   (+0.0018 vs full, SIGNIFICANT -- pruning noise helps)
chosen 4 features   0.6756   (identical to full; parsimony pick)
3 features          0.6740   (all cheap; no embedding work)
constant            0.6637
```

**42 of 46 features can be dropped with no loss.** Surviving 4, one per signal
type and with no redundancy:

| feature | cost | captures |
|---|---|---|
| `ql` | lookup | query surface (length) |
| `entropy_bm25` | scores | BM25 confidence / score dispersion |
| `apair_ratio_bm25` | **embed** | embedding coherence of BM25's top results |
| `d_wig_z` | scores | **cross-retriever** BM25-vs-dense comparison |

**Pareto choice.** `apair_ratio_bm25` is the only expensive survivor (embedding
gathers + pairwise similarity). Dropping it costs **0.0016** (0.6756 → 0.6740)
and leaves a router needing *no embedding work at all* — still **+0.0103** over
the constant. For an efficiency-led paper the 3-feature variant is arguably the
better headline, with the 4th reported as the marginal extra it buys.

### ⚠️ Caveat: the ablation is model-specific

`dev_ndcg` is **byte-identical (0.676585) for twelve consecutive rounds**
(n=41→30). Those drops changed the output *not at all* — ElasticNet's L1 penalty
had zeroed their coefficients (and/or the 10-bin quantisation absorbed the tiny
changes). So the result is **"which features the linear router uses"**, not
"which features carry signal". A tree could exploit features the linear model
zeroed.

Consequence for stage 3: **do not screen on the 4-feature set alone.** Screen
every family across several candidate sets — `{3, 4, 11, 46}` — so we can tell
whether any family benefits from more features. Otherwise the linear model's
blind spots get baked into the final router.

---

## Stage 3 — `rescreen_routers.py` (families x framings x FEATURE SETS)

Re-opens the family **and** feature-set decisions together. Two reasons:

1. The best family can change once the feature set shrinks — fewer features
   favour different inductive biases.
2. The stage-2 ablation is **model-specific** (see its caveat above): a linear
   workhorse with L1 zeroed a dozen features outright. Screening only the
   4-feature set would bake that blind spot into the final router.

So: one independent Optuna study per `(feature_set, family, framing)` over
`feature_set_sizes: [3, 4, 11, 46]`, taken from the exact surviving subsets on
the stage-2 pruning path. **If a tree on 11 features beats a linear on 4, this
is where we find out.** `catboost` and `mlp` are back in — at 3-11 features they
are cheap, and this run picks the final model.

Everything else matches stage 1 (same subsample, 80/20 fit/calibrate, dev
scoring, calibrated rule, paired bootstrap), so numbers stay comparable.

### Final pick: cheapest among the tied

The whole field has stayed inside ~0.003 across every run so far, so the nominal
maximum is noise. The rule is therefore: among configs **statistically tied**
with the nominal best (paired-bootstrap CI of the difference includes 0), take
the **fewest features**, then the **lowest inference cost class**. That is the
defensible choice *and* it is what makes the "~1 ms router" claim hold.

Cost: 96 studies x 30 trials ~= 2,880 fits. The 3/4/11-feature sets are fast;
the 46-feature set with catboost/mlp is the slow tail (~1 h of the total). Drop
`46` from `feature_set_sizes`, or catboost/mlp from `families`, if time is short
— but the 46 row is what answers "does any family want more features?".

### Outputs (`data/results/router_screening/`)

| File | Contents |
|---|---|
| `<ds>_rescreen.csv` | every trial |
| `<ds>_rescreen_best_per_config.csv` | one row per (features, family, framing) + tie test |
| `<ds>_rescreen_best.json` | **the final router**: features + family + framing + params |

### Stage-3 RESULTS (hotpotqa, dev) — 96 studies, ~2,880 fits, 73 min

Best per feature set — **more features do NOT help**:

```
f11   0.6776   <- nominal best (elasticnet|regression)
f46   0.6767   <- the FULL set is worse than 11
f3    0.6758
f4    0.6757
constant 0.6637 | oracle 0.7502
```

1. **The stage-2 blind-spot worry is resolved.** No family — tree or otherwise —
   benefited from the features the linear workhorse's L1 had zeroed. "A handful
   of features suffice" is now **model-independent**, not an artifact of how we
   pruned.
2. **Linear + calibration dominates.** 7 of the top 8 configs are
   `elasticnet`/`logreg`; every tree family ranks lower. `extra_trees` shows
   `pred_alpha_std ~= 0.036` — barely varying, drifting toward the constant.
3. **FINAL ROUTER (parsimony pick):** `f3 | logreg | binary` -> **0.6758**,
   features **`ql`, `entropy_bm25`, `d_wig_z`** (max cost class = `scores`).
   **+0.0121 over the constant** = 14% of oracle headroom. No embeddings, no
   document text, no inverted-index work: a length count, one BM25 score
   histogram, one cross-retriever score difference -> logistic regression -> bin
   lookup. Parsimony cost 0.0018 vs the nominal best (not significant).

### ⚠️ Accumulated selection bias — the main threat to validity

Dev has now been used to select three times: stage 1 (18 configs), stage 2
(~1,076 fits), stage 3 (~2,880 fits). The dev figure **0.6758 is optimistically
biased**; the honest number is the single test evaluation at stage 5. If the bias
is 0.003-0.005, the true gain over the constant is nearer +0.007-0.009 — still
positive, but the dev number must not be quoted as the headline result.

### Performance gotcha: thread oversubscription

On the shared 32-core box, `n_jobs=-1` (=32) **hangs** — profiled on 8,000x46:
`n_jobs=1` 5.8s, `4` 2.4s, **`8` 2.0s**, `16` 2.3s, `32/64/-1` stall. A single
study took ~7h before the fix and ~1 min after. The script therefore caps
`OMP/OPENBLAS/MKL_NUM_THREADS` **before importing numpy** (via `SCREEN_THREADS`,
default 8) and passes `n_jobs`/`thread_count` explicitly; `hist_gbdt` and `mlp`
take no `n_jobs` argument and rely on the env cap. Never set `n_jobs: -1` here.

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
