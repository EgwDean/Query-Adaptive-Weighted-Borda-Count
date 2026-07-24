# Router features

All 31 features are computed at query time from the query text and the two
top-`k` result lists. None requires relevance judgements, a second retrieval
pass, or an additional model call, so the router adds roughly a microsecond per
query.

Distribution features are computed over the top `features.window = 100` scores
rather than the full top-1000: the deep tail is retrieval noise and drives
several statistics systematically negative.

## Cost classes

| class | meaning |
|---|---|
| `lookup` | available directly from the query or the cached lists |
| `scores` | a statistic of the retrieval scores already returned |
| `embed` | requires the document or query embeddings |

Ties during feature ablation are broken toward dropping the more expensive
feature.

## Per-retriever score-distribution features

Computed independently for BM25 and dense, giving the suffixes `_bm25` and
`_dense`. `s` is the descending score vector, `sw` its top-`W` window.

| feature | definition | interpretation | cost |
|---|---|---|---|
| `top_score` | `s[0]` | strength of the best match | scores |
| `sigma_k` | `std(sw)` | spread of the top window | scores |
| `margin` | `s[0] - s[1]` | separation of the top result | scores |
| `norm_margin` | `(s[0] - s[1]) / (abs(s[0]) + eps)` | scale-free separation | scores |
| `smv` | mean of `(p * abs(log p))`, `p = sw / mean(sw)` | score magnitude variance | scores |
| `entropy` | entropy of `softmax(sw)` | flatness of the distribution | scores |
| `robust_sigma` | `std` of `sw` after trimming 10% from each tail | outlier-resistant spread | scores |
| `zscore_top` | `(s[0] - mean(sw)) / (std(sw) + eps)` | how far the top stands out | scores |
| `zscore_margin` | `(s[0] - s[1]) / (std(sw) + eps)` | margin relative to noise | scores |
| `autocorr` | score-weighted embedding autocorrelation over the top window | mutual consistency of the top results | embed |
| `apair_ratio` | mean pairwise similarity of top-`W` over that of bottom-`W` | tightness of the top cluster | embed |

These are query-performance-prediction statistics: a retriever that has found a
clear winner produces a different score shape from one returning undifferentiated
results.

## Cross-retriever features

| feature | definition | interpretation | cost |
|---|---|---|---|
| `jaccard` | overlap of the two top-`k` document sets | agreement on candidates | lookup |
| `kendall_tau` | Kendall tau over documents in both lists | agreement on ordering | lookup |
| `d_zscore_top` | `zscore_top_bm25 - zscore_top_dense` | which side is more confident | scores |
| `d_zscore_margin` | `zscore_margin_bm25 - zscore_margin_dense` | as above, margin form | scores |
| `d_entropy` | `entropy_bm25 - entropy_dense` | which side is more peaked | scores |
| `d_smv` | `smv_bm25 - smv_dense` | relative magnitude variance | scores |
| `d_sigma_k` | `sigma_k_bm25 - sigma_k_dense` | relative spread | scores |

The difference features state the routing question directly: whichever retriever
looks more confident on this query should receive more weight.

## Query features

| feature | definition | cost |
|---|---|---|
| `ql` | query length in whitespace tokens | lookup |
| `query_centroid_cos` | cosine between the query embedding and the corpus centroid | embed |

`query_centroid_cos` is high for generic queries close to the average of the
collection.

## Count

| group | count |
|---|---:|
| score-distribution, 11 per retriever, two retrievers | 22 |
| embedding coherence, included above (`autocorr`, `apair_ratio`) | — |
| cross-retriever differences | 5 |
| list agreement (`jaccard`, `kendall_tau`) | 2 |
| query (`ql`, `query_centroid_cos`) | 2 |
| **total** | **31** |

## Selected features

Greedy backward elimination reduces 31 to 3 with no statistically significant
loss on the development dataset. One specification is selected per fusion
function and inherited by every held-out dataset.

| fusion | features |
|---|---|
| score-minmax | `margin_bm25`, `entropy_bm25`, `smv_dense` |
| rrf | `ql`, `smv_dense`, `d_entropy` |
| borda | `ql`, `sigma_k_dense`, `d_entropy` |

All three selections are score-distribution statistics plus, for the rank
fusions, query length. Because many of the 31 features are near-duplicates, the
specific names are less meaningful than the family and the count: the useful
signal is the shape of the two score distributions, and three such statistics
capture it.

The full ablation path, with the dev NDCG@10 at every feature-set size and the
tie test against the best point, is in
`data/results/router_screening/*_ablation.csv`.
