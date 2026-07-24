# Method

## Retrievers

| | model | scores |
|---|---|---|
| lexical | BM25 (Lucene variant), `k1=0.8`, `b=0.4`, Snowball stemming | unbounded, positive |
| semantic | `sentence-transformers/all-mpnet-base-v2`, 768-d, `max_seq_length=384`, L2-normalised | cosine, in `[-1, 1]` |

Each retriever returns the top `top_k = 1000` documents per query. Both the ranked
identifiers and the raw scores are cached, because the fusion functions below need
the magnitudes, not only the order.

## Normalisation

Raw BM25 and cosine scores are not comparable, so each retriever's scores are
min-max normalised per query over its own top-k:

```
norm(s) = (s - min(s)) / (max(s) - min(s))          in [0, 1]
```

If every score in a list is identical the list normalises to zero. A document
retrieved by only one retriever contributes zero from the other side.

## Fusion functions

`alpha` is the fusion weight: `alpha = 1` is pure lexical, `alpha = 0` pure
semantic. Score fusion is the primary method; the two rank fusions are baselines.

**Score fusion (primary)**

```
fuse(d) = alpha * norm(bm25_score(d)) + (1 - alpha) * norm(dense_score(d))
```

**Reciprocal rank fusion**

```
rrf(d) = alpha * 1/(K + rank_bm25(d)) + (1 - alpha) * 1/(K + rank_dense(d))      K = 60
```

**Borda count**

```
borda(d) = alpha * (N - rank_bm25(d)) + (1 - alpha) * (N - rank_dense(d))        N = 1000
```

Ranks are 0-based internally. A document absent from a list scores zero on that
side, which is below the value of being ranked last.

Score fusion is primary because rank fusion discards score magnitude. That
comparison is established in Bruch, Gai & Ingber, *An Analysis of Fusion
Functions for Hybrid Retrieval*, ACM TOIS 2023, and is not a claim of this work.
RRF and Borda are carried through the whole study so that every finding can be
checked for dependence on the fusion function.

A document is treated as retrieved by a side only if that side actually scored
it. Empty-query fallback rows and padded short result lists carry a raw score of
zero and are dropped, which leaves score fusion unchanged and prevents the rank
fusions from awarding rank points to positions no retriever scored.

## Oracle alpha and the alpha-NDCG curve

For every query, NDCG@10 is evaluated at all 101 grid points
`alpha = 0.00, 0.01, ..., 1.00`, and the whole curve is stored.

| quantity | definition |
|---|---|
| `alpha` (oracle) | `argmax` over the grid of the query's NDCG@10; ties resolve to the lowest alpha |
| `oracle_ndcg` | the corresponding maximum |
| `alpha_sensitivity` | `max(curve) - min(curve)` |
| `plateau_frac` | fraction of grid points within 1e-6 of the maximum |

Storing the curve rather than only its argmax means any later predicted alpha is
scored by table lookup instead of re-running retrieval. `alpha_sensitivity` is
used as a training sample weight: a query whose curve is flat has an essentially
arbitrary oracle label and is down-weighted.

The oracle uses relevance judgements to choose alpha and is therefore an upper
bound, not a method.

## Headroom

```
static_best = NDCG@10 of the single best constant alpha, tuned on the non-test split
oracle      = mean over queries of each query's own best NDCG@10
headroom    = oracle - static_best
gain        = router - static_best
```

The baseline is a properly tuned constant. Comparison against `alpha = 0.5` is
reported for context only.

## Router

The router predicts alpha from 31 features computed from the query text and the
two result lists (see [FEATURES.md](FEATURES.md)). It never sees relevance
judgements at inference time.

Three label framings are screened for every model family:

| framing | target | model output |
|---|---|---|
| `regression` | oracle alpha | predicted alpha |
| `binary` | `alpha > 0.5` | probability |
| `multibin` | nearest of 11 alpha bins | expected alpha over bins |

Families screened: LightGBM, XGBoost, CatBoost, HistGradientBoosting,
RandomForest, ExtraTrees, ElasticNet, LogisticRegression, MLP. SVM and k-NN are
excluded on inference cost: SVM training is superlinear in dataset size and k-NN
pays a corpus-sized cost per query.

Hyperparameters are tuned by an independent Optuna TPE study per
(family, framing) pair, 30 trials each, maximising mean NDCG@10 on the selection
split. Features are then pruned by greedy backward elimination, and the smallest
feature set statistically tied with the best point is kept. Backward elimination
is used because redundant features mask each other under importance ranking.

## Decision rule

The framing determines what the model predicts. The decision rule determines how
that prediction becomes an alpha. The two are independent.

**raw** uses the model output directly as alpha. A regressor fitted to oracle
alpha labels estimates the conditional mean of alpha, and model outputs are
typically compressed into a narrow band. The alpha that maximises mean NDCG is
systematically higher than the mean oracle alpha, so the raw rule under-weights
the lexical retriever.

**calibrated** (histogram binning) uses the model only to rank queries:

1. sort queries by model output and cut into `n_calib_bins` quantile bins,
2. average the alpha-NDCG curves of the queries in each bin,
3. the bin emits the alpha maximising its own averaged curve.

The bin-to-alpha table is fitted on a held-out slice of the training split
(`calib_fraction = 0.2`) and frozen. At inference the router predicts, locates
the bin, and returns the stored alpha.

### What the safety property does and does not guarantee

On the data the bins are fitted on, the calibrated rule cannot underperform the
best constant. For each bin `b` with empirical mean curve `C_b`, the emitted
alpha satisfies `C_b(alpha_b) >= C_b(alpha*)` by definition of `argmax`, and
summing over bins gives the result.

This is an in-sample guarantee. On unseen queries each `alpha_b` is a noisy
estimate, and because `alpha*` maximises the true mean curve, any deviation costs
NDCG. With a signal-free model the calibrated rule is therefore expected to be
marginally *worse* out of sample, by an amount that grows with the curvature of
the mean curve near its peak and with the variance of the per-bin estimate.

Measured on this study's curves, a bin that lands 0.05 away from `alpha*` costs
0.0005 to 0.0059 NDCG, and 0.10 away costs 0.004 to 0.020 — the same order as the
gains being measured. Empirically, across 126 configurations the calibrated rule
was significantly worse than the constant in 1 case, and significantly better in
58.

`MIN_QUERIES_PER_BIN = 50` caps the bin count so that each bin retains enough
queries to estimate its alpha, which bounds the variance term:

```
n_bins = max(1, min(requested_bins, n_calibration_queries // 50))
```

At one bin the rule degenerates to the constant baseline. On scifact this cap
changed a significant loss of -0.044 into a non-significant -0.003.

## Metrics

NDCG@10 is the primary metric, with gain `2^rel - 1` and log2 discount, computed
over the fused top-10. NDCG@100, MRR@100 and Recall@100 are reported alongside.
Confidence intervals and significance come from a paired bootstrap over queries
(1000 resamples); see [PROTOCOL.md](PROTOCOL.md).
