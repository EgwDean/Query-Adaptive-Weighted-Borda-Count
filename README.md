# Query-Adaptive Score Fusion in Hybrid Retrieval

A study of when a per-query fusion weight helps in hybrid lexical/semantic
retrieval, why it usually does not, and what is required to deploy it safely.

Two retrievers — BM25 and `all-mpnet-base-v2` — are combined by a convex
combination of per-query min-max normalised scores:

```
fuse(d) = alpha * norm(bm25_score(d)) + (1 - alpha) * norm(dense_score(d))
```

`alpha = 1` is pure lexical, `alpha = 0` pure semantic. A router predicts `alpha`
per query. The question is whether that beats the best single global `alpha`.

Score fusion is the primary method. RRF and Borda appear only as baselines: that
rank fusion discards score magnitude is established in Bruch, Gai & Ingber, ACM
TOIS 2023, and is not claimed here. Both are carried through the full study so
every finding can be tested for dependence on the fusion function.

## Findings

The study covers **7 BEIR datasets x 3 fusion functions = 21 cells**, plus 252
decision-rule runs. `hotpotqa` is the development dataset; the other six are
held out and inherit its frozen router specification, refitting only weights and
the calibration table.

**A router's raw output is not a fusion weight.** Across 126 configurations
spanning 7 datasets, using the model output directly as `alpha` was significantly
worse than a tuned constant in **55** cases and significantly better in **3**.
Replacing the decision rule with histogram-binning calibration, holding the model
fixed, gives **58** significant improvements and **1** significant loss.

**The cause is measurable.** A model fitted to oracle-`alpha` labels estimates the
conditional mean of `alpha`, but the `alpha` that maximises mean NDCG is
systematically higher. Raw predictions fell below the NDCG-optimal `alpha` on
**7 of 7** datasets, mean gap −0.19. Choosing a constant by averaging oracle
`alpha` values instead of maximising NDCG costs up to 0.126 NDCG, an order of
magnitude more than the adaptive gain itself.

**Gains track retriever complementarity.** Measuring complementarity as the
interquartile range of the oracle-`alpha` distribution, the dataset-level rank
correlation with realised gain is **Spearman rho = +0.943, p = 0.005** over the 6
independent held-out datasets. Complementarity is necessary but not sufficient:
`msmarco` has 6,980 evaluation queries and ample statistical power, but low
spread, and gains only +0.0027.

**The pattern is fusion-invariant.** The same behaviour appears under score
fusion, RRF and Borda.

### Score fusion, test split

| dataset | oracle-α IQR | queries | constant α* | router | oracle | gain | significant |
|---|---:|---:|---:|---:|---:|---:|:--:|
| hotpotqa (dev) | 0.57 | 7,405 | 0.6747 | 0.6810 | 0.7238 | +0.0063 | yes |
| fever | 0.44 | 6,666 | 0.7285 | 0.7375 | 0.8057 | +0.0089 | yes |
| msmarco | 0.14 | 6,980 | 0.4123 | 0.4150 | 0.5194 | +0.0027 | yes |
| nfcorpus | 0.45 | 323 | 0.3743 | 0.3727 | 0.4211 | −0.0016 | no |
| scifact | 0.12 | 300 | 0.7324 | 0.7289 | 0.7885 | −0.0035 | no |
| fiqa | 0.08 | 648 | 0.5128 | 0.5146 | 0.5787 | +0.0017 | no |
| quora | 0.00 | 10,000 | 0.9013 | 0.8986 | 0.9346 | −0.0027 | yes |

`gain` is router minus the tuned constant `alpha*`; significance is a paired
bootstrap of the per-query NDCG@10 difference on test, 95% CI excluding zero.
`quora` is the pre-registered negative control: with zero complementarity there
is nothing to route on, and routing costs a small but detectable amount.

Absolute gains are small. That is the finding, not a limitation of the
implementation: the oracle ceiling itself sits close to the tuned constant on
most collections.

## Repository layout

```
config.yaml              all settings, documented inline
src/
  pipeline.py            entry point; sections 0-3 (download, embed, tune, retrieve)
  sections.py            sections 4-9 (features, screening, ablation, fit, benchmark)
  core.py                BEIR I/O, metrics, fusion functions, alpha curve, bootstrap
  utils.py               config and path helpers
  run_study.py           runs the full dataset x fusion matrix
  h2_decision_rule.py    raw versus calibrated decision-rule experiment
  probe_datasets.py      reports split sizes for candidate datasets
docs/
  METHOD.md              fusion maths, oracle, router, calibration
  PROTOCOL.md            datasets, splits, inheritance, significance, limitations
  FEATURES.md            the 31 router features
  REPRODUCING.md         environment, commands, cost, verification
data/results/            published result tables (see below)
```

## Published results

Large regenerable intermediates (feature tables, curve arrays, serialised
estimators, logs) are not tracked. What is published is sufficient to check every
reported number:

| path | contents |
|---|---|
| `router_final/STUDY_SUMMARY.csv` | one row per cell: IQR, baseline, router, oracle, gain, CI, significance |
| `router_final/*_benchmark.csv` | all methods per cell, NDCG@10/@100, MRR@100, Recall@100, CIs |
| `router_final/*_benchmark_per_query.csv` | per-query NDCG@10 for every method |
| `router_final/*_router_meta.json` | the frozen router specification per cell |
| `router_final/h2_decision_rule_ALL.csv` | raw versus calibrated across all datasets |
| `router_screening/*.csv` | screening, ablation and re-screening paths on the development dataset |

Confidence intervals can be recomputed from the per-query files without running
the pipeline; see [docs/REPRODUCING.md](docs/REPRODUCING.md).

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python src/run_study.py                      # full matrix, resumable
python src/h2_decision_rule.py --datasets all # decision-rule experiment
```

Full details, including cost and how to add a dataset, are in
[docs/REPRODUCING.md](docs/REPRODUCING.md).

## Limitations

Stated in full in [docs/PROTOCOL.md](docs/PROTOCOL.md). In brief: BM25
hyperparameters were tuned once on the development dataset and inherited; one
retriever pair was used; the complementarity result rests on 6 independent
held-out datasets; intervals are per cell with no multiple-comparison
correction; and the calibration safety property is an in-sample guarantee that
can be violated marginally out of sample.
