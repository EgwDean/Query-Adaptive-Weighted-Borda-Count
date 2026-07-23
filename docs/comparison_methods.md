# Comparison Methods & Baselines

> **Updated for convex score fusion.** The primary baseline is now
> *static score fusion with one global alpha*. Rank fusion (RRF, Borda) are
> standard baselines only — that score fusion beats them is settled in the
> literature (Bruch et al., TOIS 2023), not a contribution here.

Reference for the evaluation baselines used in this study.

**Golden rule:** every fusion method consumes the **same inputs** — the same
tuned BM25 top-k and the same all-mpnet-base-v2 dense top-k — so any difference
is attributable to the *fusion rule*, not the ingredients.

**Primary comparison:** adaptive score fusion (Tier 2) vs **globally-tuned
static score fusion** (Tier 1). Beating `α=0.5` is not a result; beating the
best single global `α` — on the SAME fusion function — is.

---

## Methods

| # | Method | Tier | Paradigm | Inputs / stage | Role | Tuning |
|---|--------|------|----------|----------------|------|--------|
| 1 | BM25 (tuned k1/b) | 0 | Sparse / lexical | single retriever | Floor | grid k1,b on train |
| 2 | Dense (all-mpnet-base-v2) | 0 | Dense / semantic | single retriever | Floor | — |
| 3 | Score fusion, α=0.5 | 1 | Fusion (score) | BM25 + dense | Baseline (naïve) | none |
| 4 | **Static score fusion, global α\*** | 1 | Fusion (score) | BM25 + dense | **Primary baseline** | single α\* on dev |
| 5 | RRF (untuned, k=60) | 1 | Fusion (rank) | BM25 + dense | Baseline (field standard) | none |
| 6 | Weighted RRF, global α\* | 1 | Fusion (rank) | BM25 + dense | Baseline | single α\* on train |
| 7 | Static Borda, global α\* | 1 | Fusion (rank) | BM25 + dense | Baseline | single α\* on dev |
| 8 | **Query-Adaptive Score Fusion** | 2 | Adaptive fusion | BM25 + dense + router | **Proposed method** | router trained on train |
| 9 | SPLADE | 3 | Learned sparse | single retriever / alt sparse leg | Topline (context) | — |
| 10 | Cross-encoder reranker | 3 | Reranking (2nd stage) | rerank top-k of a 1st stage | Topline (context) | on BM25 / dense / fusion |
| 11 | Oracle α (per-query best) | — | Adaptive fusion | BM25 + dense + qrels | Upper bound / ceiling | per-query brute force |

**Tier legend** — 0: single-retriever floors · 1: static-fusion baselines (the
real competition) · 2: proposed adaptive method + ablations · 3: stronger /
extra-stage context (report separately; changes the ingredients or adds a
stage — apples-to-oranges with single-stage fusion).

---

## Evaluation protocol notes

- **Same candidate pool** for all fusion rows (union of BM25 ∪ dense top-k);
  keep k consistent (≥ 100, consider 1000 for the recall ceiling).
- **Metrics:** NDCG@{10,100}, MRR@100, Recall@100. Report **cost** (latency /
  router overhead) next to quality — the selling point is "near-topline quality
  at fusion cost," which only lands with a cost column.
- **Significance:** paired two-sided t-test + 95% bootstrap CIs on per-query
  scores; report Cohen's d. The decisive test is **#8 vs #4**.
- **Held-out test set** only; select the dataset and tune all α / k1,b / router
  hyperparameters on train (or train+dev) — never on test.
- **Tier 3 caveat:** the cross-encoder barely helped (and hurt dense) on small
  BEIR sets but helps at MS MARCO scale — report it as an add-on layer, not a
  single-stage baseline.
