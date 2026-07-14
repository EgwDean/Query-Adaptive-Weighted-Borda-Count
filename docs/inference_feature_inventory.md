# Inference-Time Feature Inventory (Phase 2 dataset plan)

Which features from [ltr_router_features.md](ltr_router_features.md) can
actually be computed **at inference time** (query text + retrieval results
only — no qrels), and how we'd compute each one for the planned Phase-2
dataset: one row per hotpotqa query, columns = these features, label = oracle
alpha (from `alpha_distribution.py`). **This is a plan only — no dataset is
built yet.**

**Rule for inclusion:** a feature is *available* if it can be computed from
(a) the query text, (b) corpus-wide statistics computed once from the corpus
(no relevance judgments), and/or (c) the two retrieved top-k lists (BM25 +
dense) and their raw scores — i.e. everything the *live system* would have
when a new, unlabeled query arrives. Anything needing qrels (e.g. `F_q`, the
rank of the first *relevant* doc) is a **label**, not a feature — excluded
here, already covered in `ltr_router_features.md`.

**Key design decision:** most post-retrieval features are defined for *one*
ranked list. For routing, compute each **twice** — once on the BM25 list,
once on the dense list (suffix `_bm25` / `_dense`) — and add the **difference
and/or ratio** as an explicit derived feature (e.g. `nqc_bm25 - nqc_dense`).
The comparison between the two sides, not either side alone, is what should
carry the routing signal.

---

## One-time corpus-level assets (build once, reuse for every query)

These are computed once per dataset (hotpotqa) and cached, so per-query
feature computation is cheap:

| Asset | What it is | How to build |
|---|---|---|
| `N`, `f_t` (doc frequency per term) | Needed for all IDF-family features | Already inside the `bm25s` index built by `alpha_distribution.py` / `tune_bm25.py` — no extra pass needed, just expose it. |
| `tf_coll(t)`, `tokens_coll` | Total occurrences of each term in the whole collection, and total token count | One pass over the already-tokenized corpus (reuse the tokens `bm25s.tokenize` produces) building a `Counter`; save as a `term -> count` table (parquet/pickle). |
| BM25 term-document sparse matrix | Per-doc term weights `tf_{t,d}` for every term/doc pair | Already built internally by `bm25s` for scoring — reuse it directly instead of re-scanning, for VAR(t), query-scope, and co-occurrence features. |
| Corpus dense embeddings | `corpus_emb.npy` (one row per doc) | Already produced by `embed.py` — reused as-is; any doc's embedding is a memmap lookup by row index. |
| Collection embedding centroid | Mean of all `corpus_emb` rows (single 768-dim vector) | One pass over `corpus_emb.npy` (memmap-friendly, can be computed with a running mean while streaming shards); saved once. |
| BM25 "whole corpus as one document" score baseline | `score(Q, Corpus)` for WIG/NQC, lexical side | Formula using `tf_coll(t)` and `tokens_coll` (collection-wide term frequency in place of per-doc term frequency) — computed once per query at inference from the term table above, no re-indexing needed. |

---

## A. Query-only features (pre-retrieval; no retrieval run needed)

All available. All derive from the term-statistics asset above; per-query cost
is a single pass over the query's own (few) tokens.

| Feature | Available? | How computed |
|---|---|---|
| Query length (ql) | Yes | Count non-stopword tokens in the query. |
| Average IDF (avgIDF) | Yes | Mean of `log(N/f_t)` over query terms, using the cached `f_t` table. |
| Max IDF (maxIDF) | Yes | Max of the same per-term idf values. |
| Std. dev. of IDF (γ1) | Yes | Std. dev. of the same per-term idf values. |
| Max/min IDF ratio (γ2) | Yes | `idf_max / idf_min` from the same values. |
| Query scope (ω) | Yes | `n_Q` = size of the union of posting lists (docs containing >=1 query term); get via the BM25 sparse term-document matrix (union of the rows/columns for the query's terms — cheap since a query has few terms). |
| Simplified Clarity Score (SCS) | Yes | Query term frequencies (`qtf/ql`) vs. `P_coll(w) = tf_coll(w)/tokens_coll` from the term table. |
| AvICTF | Yes | `tf_coll(t)` + `tokens_coll` from the term table, per query term, averaged. |
| SCQ per term, Sum/Avg/MaxSCQ | Yes | `tf_coll(t)` (collection freq) + `f_t` (doc freq), both cached — pure lookup + formula per query term. |
| Term weight variance VAR(t), Sum/Avg/MaxVAR | Yes | In principle a per-*term* statistic (doesn't depend on the query beyond which terms it contains) — can be **precomputed once for the full vocabulary** from the BM25 sparse matrix (variance of `w_{t,d}` across all docs containing t) and cached as a `term -> VAR(t)` lookup table, making it a cheap per-query lookup, not a per-query scan. |
| Query term co-occurrence (PMI) | Yes | Intersect posting lists (from the BM25 sparse matrix) for each pair of query terms; a query has few terms so few pairs — cheap per query. |
| Embedding-based query specificity | Partial | The doc describes *word*-embedding pairwise similarity, which needs token-level embeddings we don't have (mpnet is a sentence encoder). Practical substitute: cosine of the **whole-query embedding** (already computed by `embed.py`) to the **collection centroid** asset above — a legitimate specificity proxy, cheap (one dot product), but not identical to the literature's exact formula. |

## B. Per-retriever post-retrieval features (compute once per list: BM25 and dense)

All of these need only the retrieved top-k **scores** for that query — note
the current `bm25_retrieve`/`dense_retrieve` in `alpha_distribution.py`
discard raw scores and keep only ranked doc-id order; the Phase-2 dataset
build will need to also capture the scores.

| Feature | Available? | How computed |
|---|---|---|
| Top (max) retrieval score | Yes | The retriever's own top-1 score, per list — trivial. |
| Std. dev. of top-k scores (σ_k) | Yes | Std. dev. of the top-k score list — trivial. |
| Score margin (top-1 minus top-2) | Yes | Not in the original catalogue as a named row, but implied by "Retriever Confidence" style features elsewhere — trivial, cheap, and one of the strongest routing signals per the hybrid-routing literature. |
| Weighted Information Gain (WIG) | Yes | Top-k scores minus the `score(Q,Corpus)` baseline asset (computed per side — BM25's own baseline formula for the lexical list; cosine of query embedding to the centroid asset, scaled comparably, for the dense list). |
| Normalized Query Commitment (NQC) | Yes | `σ_k / abs(score(Q,Corpus))`, same baseline asset as WIG. |
| Score Magnitude and Variance (SMV) | Yes | Function of top-k scores and their mean — trivial once top-k scores are kept. |
| Entropy of score distribution | Yes | Softmax the top-k scores, compute entropy — trivial. |
| Robust standard deviation estimation | Yes | Trimmed σ_k over the top-k scores — trivial. |
| Clarity Score (CS) | Yes, moderate cost | Needs a query language model built from the *text* of the top-k retrieved docs, compared to the collection LM (`tf_coll` asset) — requires reading the top-k docs' text at query time (they're already fetched for retrieval), so no extra corpus scan, just extra per-query text processing. |
| Score autocorrelation (cluster-hypothesis) | Yes, cheap | We already have **every** doc's dense embedding cached (`corpus_emb.npy`), regardless of which retriever surfaced it — so a doc-similarity graph over the top-k list (from *either* retriever) is a cheap lookup + cosine matrix, no extra encoding needed. |
| Query Feedback (QF) | Yes, but costly | Needs one extra retrieval pass per side (expand the query from its own top-k terms, re-retrieve, measure overlap) — doable but roughly doubles per-query retrieval cost. Recommend: optional / second pass, not in the v1 feature set. |
| Ranking Robustness | Yes, but costly | Needs multiple extra retrieval passes with perturbed embeddings/representations — meaningfully more expensive. Recommend: skip for v1. |
| Utility Estimation Framework (UEF) | Yes, but costly | Built on top of RM-feedback re-ranking (same machinery as Query Feedback) — same cost caveat; skip for v1. |
| Reference-list based estimation | Yes, but heavier setup | Needs a bank of "reference queries" with known effectiveness — usable here since we'll have a **train split with computed oracle alpha/NDCG** to serve as that reference bank, but it's a training-set asset, not a pure corpus asset, and adds pipeline complexity. Recommend: optional / v2. |

## C. Dense-embedding-specific features

All available and cheap, since `embed.py` already caches an embedding for
**every** corpus document (not just the ones the dense retriever itself
surfaced) plus every query.

| Feature | Available? | How computed |
|---|---|---|
| Embedding-based query specificity (dense) | Yes | Cosine of the query embedding to its nearest doc embeddings — this is effectively already produced as the dense retriever's own top-k scores (cosine similarities), so it's largely redundant with "top dense score" / "dense score margin" above rather than a separate computation. |
| Coherence-based dense predictor (A-Pair-Ratio) | Yes | Avg. pairwise cosine among the top-k retrieved docs' cached embeddings, divided by the same among a "bottom-k" sample (e.g. ranks 90-100 of the same list) — cheap, reuses cached embeddings, computable for **either** list (BM25 or dense) since embeddings exist for all docs. |

## D. Hybrid lexical/semantic routing–specific features

| Feature | Available? | How computed |
|---|---|---|
| Rank-list agreement (Jaccard@k / Kendall's τ) | Yes | Directly from the two top-k doc-id lists already produced by `bm25_retrieve` / `dense_retrieve` — no extra cost. One of the most directly relevant features per the literature precedent. |
| Sparse–dense score margin (DAT-style) | Partial | The original method needs an **LLM judge** per query (real inference cost + external dependency) to rate top-1 effectiveness. Cheap substitute for v1: use the retrievers' **own normalized top-1 scores** directly (min-max or z-score normalized BM25 vs. dense score) instead of an LLM judgment — loses DAT's "true effectiveness" grounding but costs nothing extra and reuses data we already have. Flag as an approximation, not the literal DAT feature. |
| Query-only cross-encoder routing score | No (not a feature) | This is a full alternative **model architecture** (a trained classifier over the query alone), not a scalar input to add to a tabular dataset — it would replace the hand-crafted-feature approach entirely, not extend it. Worth considering later as an end-to-end baseline, out of scope for this feature table. |

---

## Excluded entirely (not usable as inference-time tabular features)

| Feature | Why excluded |
|---|---|
| Sparse-retriever rank of first relevant doc (`F_q`) | Needs qrels — this is the **label**, not a feature (same role as our own oracle alpha). |
| BERT-QPP score | Needs its own trained regressor as a preprocessing step, not a hand-computed number — a candidate meta-feature for a v2 ensemble, not v1. |
| NeuralQPP (learned combination) | By definition a trained combiner *over* other QPP features — redundant with training our own router directly on the features below; not a separate input. |
| Relative Information Gain (RIG) | Needs automatically generated query variants (paraphrasing/back-translation), a nontrivial extra generation step — v2 candidate, not v1. |

---

## Recommended v1 feature set for the hotpotqa dataset build

Putting the above together, a practical first cut (cheap, all "Yes" rows
above, no extra retrieval passes):

- **Query-only (group A):** ql, avgIDF, maxIDF, γ1, γ2, ω, SCS, AvICTF,
  AvgSCQ/MaxSCQ, AvgVAR/MaxVAR, avg PMI co-occurrence, query-centroid cosine.
- **Per retriever, computed twice (`_bm25`, `_dense`) + a derived diff/ratio
  (group B/C):** top-1 score, score margin (top1-top2), σ_k, WIG, NQC, SMV,
  entropy, robust σ, Clarity Score, A-Pair-Ratio, score-autocorrelation.
- **Cross-retriever (group D):** Jaccard@k, Kendall's τ, normalized
  score-margin (DAT-style proxy).

Deferred to a later pass if the v1 model needs more signal: Query Feedback,
Ranking Robustness, UEF, Reference-list estimation (all costly — extra
retrieval passes), BERT-QPP/NeuralQPP/RIG (all need their own sub-models or
generation step).
