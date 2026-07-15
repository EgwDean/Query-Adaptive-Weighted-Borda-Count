# Phase-2 Feature Dataset (`src/create_dataset.py`)

The dataset used to train a regressor that predicts the **optimal fusion weight
`alpha`** per query — i.e. how lexical vs. semantic a query is — so Weighted
Borda Count can be applied adaptively at inference time.

- **One CSV per split** (train / dev / test):
  `data/results/feature_dataset/<dataset>_<split>_features.csv`
- **One row per query.** Columns = the inference-time features below.
- **Label = `alpha`** (the oracle fusion weight), recomputed *fresh* in this
  script at `retrieval.top_k` / `retrieval.eval_k` so features and label come
  from the exact same retrieval configuration.

Every feature is an **inference-time** signal: computable from the query text,
one-time corpus statistics, and the two retrieved top-k lists (+ their raw
scores). Nothing here needs relevance judgments. See
[ltr_router_features.md](ltr_router_features.md) for the wider catalogue and
[inference_feature_inventory.md](inference_feature_inventory.md) for the
availability analysis this selection came from.

---

## How it runs

1. **Corpus assets, built once and cached** under
   `data/processed_data/<dataset>/assets/` (reused across all splits):
   term-document count matrix (inverted index), `df` / `cf` / `idf` / `VAR(t)`
   tables, `tokens_coll`, and the collection embedding centroid. Corpus
   embeddings (`corpus_emb.npy`) come from `embed.py` and are reused as-is.
2. **Per split:** query embeddings are computed **in-script** (no separate
   `embed.py` run per split), both retrievers run to depth `top_k` **keeping raw
   scores**, features + oracle `alpha` are computed in one pass, and the CSV is
   written. Re-running **skips** any split whose CSV already exists (resumable).

Prerequisite: `python src/embed.py` for the active dataset (produces
`corpus_emb.npy` / `corpus_ids.json`). Toggle any feature group off in
`config.yaml` → `create_dataset.features` for ablation.

---

## Feature dictionary

Notation: `|Q|`/`ql` = query token count; `N` = #docs; `f_t`/`df` = doc
frequency; `cf` = collection frequency; `tokens_coll` = total tokens;
`idf(t) = log(N/df_t)`; `s` = a retriever's top-k **raw** score vector (desc),
`mu`/`sd` its mean/std.

### Metadata & label
| Column | Meaning |
|---|---|
| `dataset`, `split`, `qid` | identifiers |
| **`alpha`** | **LABEL** — oracle fusion weight maximising NDCG@`eval_k` (1 = BM25, 0 = dense; lowest wins ties) |
| `oracle_ndcg`, `bm25_ndcg`, `dense_ndcg` | reference NDCG@`eval_k` for the fused / pure-BM25 / pure-dense rankings |
| `n_rel`, `eval_k`, `top_k` | #relevant docs, scoring cutoff, pool depth |

### A. Query-only (`features.query_only`)
| Column | Computation |
|---|---|
| `ql` | number of query tokens (after stopword removal + stemming) |
| `avg_idf`, `max_idf`, `std_idf` | mean / max / std of `idf(t)` over query terms |
| `idf_ratio` | `max_idf / min_idf` (γ2) |
| `scs` | Simplified Clarity Score: `Σ P_ml(w|Q)·log2(P_ml/P_coll)`, `P_ml=qtf/ql`, `P_coll=cf/tokens_coll` |
| `avictf` | mean of `log2(tokens_coll/cf_t)` over query terms |
| `scq_sum`, `scq_avg`, `scq_max` | `SCQ(t)=(1+ln cf_t)·ln(1+N/df_t)` aggregated |
| `var_sum`, `var_avg`, `var_max` | `VAR(t)` aggregated (see note) |
| `query_centroid_cos` | cosine(query embedding, collection centroid) — embedding-based query specificity |
| `query_scope` | `-log(n_Q/N)`, `n_Q` = #docs containing ≥1 query term (`features.query_scope`) |
| `pmi_avg` | mean pairwise PMI of query terms from co-occurrence counts (`features.pmi`) |

### B/C. Per-retriever score distribution (`features.per_retriever_scores`)
Computed **twice**, suffix `_bm25` and `_dense`. Distribution-shape stats
(`sigma_k`, `wig`, `nqc`, `smv`, `entropy`, `robust_sigma`) use the top
`score_window_k` scores (default 100); the deep top_k tail is retrieval noise
that dilutes them (e.g. it drives WIG systematically negative). `top_score` and
`margin` still come from the very top; retrieval/fusion still use the full
`top_k` pool.

| Column | Computation |
|---|---|
| `top_score_{r}` | `s[0]` |
| `sigma_k_{r}` | std of top-k scores (σ_k) |
| `margin_{r}` | `s[0]-s[1]` |
| `norm_margin_{r}` | `(s[0]-s[1])/|s[0]|` (scale-free) |
| `wig_{r}` | `mean(s) - baseline` (Weighted Information Gain) |
| `nqc_{r}` | `sd / |baseline|` (Normalized Query Commitment) |
| `smv_{r}` | `mean( (s/μ)·|ln(s/μ)| )` (Score Magnitude & Variance) |
| `entropy_{r}` | entropy of `softmax(s)` |
| `robust_sigma_{r}` | std of top-k scores after 10% trimming |
| `autocorr_{r}` | Moran's-I score autocorrelation over the top-W doc-similarity graph (`features.coherence`) |
| `apair_ratio_{r}` | mean pairwise cosine of top-W docs ÷ that of bottom-W docs (`features.coherence`) |

**Baselines** (the `score(Q,Corpus)` term): BM25 side = BM25 score of the query
against the whole corpus treated as one document (from `cf`, `tokens_coll`,
`avgdl`); dense side = cosine(query embedding, collection centroid).

### Clarity Score (`features.clarity`)
| Column | Computation |
|---|---|
| `clarity_bm25`, `clarity_dense` | KL of a query LM (built from the top-`clarity_feedback_k` retrieved docs, weighted by `softmax(score)`) vs. the collection LM |

### D. Cross-retriever (`features.cross_retriever`)
| Column | Computation |
|---|---|
| `jaccard` | Jaccard overlap of the two top-k lists |
| `kendall_tau` | Kendall's τ over docs in both lists (by rank) |
| `d_ztop` | z-standardised top score: `(s[0]-μ)/sd`, BM25 minus dense |
| `d_zmargin` | z-standardised margin `(s[0]-s[1])/sd`, BM25 minus dense |
| `d_zentropy` | entropy of `softmax(z)` (z-standardised scores), BM25 minus dense |
| `d_wig_z` | `(μ-baseline)/sd`, BM25 minus dense |

**Cross-retriever scale-invariance = z-score.** BM25 scores and cosine
similarities are incommensurable, so before any cross-retriever *difference*
each retriever's top-`score_window_k` score vector is standardised to mean 0 /
std 1 per query (`z = (s-μ)/sd`); the `d_*` features difference those
standardised quantities. `jaccard`/`kendall_tau` use the full top_k lists.

---

## Implementation notes / deviations from the literature
- **`VAR(t)`** uses `w_{t,d}=(1+ln tf_{t,d})·idf(t)` over docs containing `t`,
  computed from the cached count matrix — a tf-idf weighting, not BM25's own.
- **WIG/NQC** were defined for language-model (log-prob) retrieval; porting the
  "corpus-as-one-document" baseline to BM25 (and a centroid baseline to dense)
  is a documented adaptation, not the verbatim source formula.
- **Clarity** is computed over a **capped** feedback set (`clarity_feedback_k`,
  default 50), standard practice and cheap since those docs are already fetched.
- **Coherence** features use a **top-W / bottom-W window** (`coherence_window`,
  default 100) rather than the full 1000 — the 1000-tail is mostly noise and it
  keeps the O(W²) similarity matrix trivially cheap (one small matmul).
- OOV query terms (not in the corpus vocab) are dropped from group-A features.

---

## The road here (project history)

1. **Idea.** Query-adaptive hybrid retrieval fusing BM25 (lexical) + dense
   (semantic) with **Weighted Borda Count**, weight `alpha` learned per query.
2. **Parity choice.** Dense model = **all-mpnet-base-v2**, picked for *strength
   parity* with tuned BM25 (a stronger model collapses the oracle-`alpha`
   distribution toward 0 and leaves nothing to route). Apache-2.0, free for
   research/commercial use.
3. **Phase 1 — dataset selection.** `download.py` → `embed.py` →
   `alpha_distribution.py` computed the oracle-`alpha` distribution across 14
   BEIR datasets (memmap + chunked retrieval so MS-MARCO-scale corpora fit on a
   24 GB GPU). Datasets ranked by `alpha` spread; **hotpotqa** chosen.
4. **Metric + depth.** Selection used NDCG@100 (consistently across datasets);
   the project then switched to **NDCG@10 (primary)** and decoupled
   `retrieval.top_k` (candidate/Borda pool) from `retrieval.eval_k` (scoring
   cutoff). Pool depth raised **100 → 1000** to match standard BEIR practice
   (retrieve deep, evaluate shallow). See
   [bm25_parameter_history.md](bm25_parameter_history.md).
5. **BM25 tuning.** `tune_bm25.py` grid-searches `k1`/`b`/stemming per corpus by
   mean NDCG@`eval_k`; hotpotqa → `k1=0.8, b=0.4, stemming=on`.
6. **Feature design.** From the QPP + hybrid-routing literature
   ([ltr_router_features.md](ltr_router_features.md)) we kept the **lightweight,
   inference-time** signals and dropped the expensive ones (LLM judges, Query
   Feedback, Ranking Robustness, UEF, reference-list — all need extra retrieval
   passes / external calls). Availability analysis in
   [inference_feature_inventory.md](inference_feature_inventory.md).
7. **This dataset.** `create_dataset.py` builds corpus assets once, then per
   split computes the features above + a fresh oracle-`alpha` label. The
   resulting tables train the `alpha` regressor; features are pruned later via
   an **ablation study** (hence the config toggles and deliberately wide set).

**Next:** train the `alpha` regressor on `*_train_features.csv`, tune on dev,
evaluate on test, and compare Weighted Borda Count against the baselines in
[comparison_methods.md](comparison_methods.md) — above all a **globally-tuned
static `alpha`**.
