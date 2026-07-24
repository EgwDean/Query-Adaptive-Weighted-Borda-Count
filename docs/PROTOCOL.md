# Experimental protocol

## Datasets

Seven BEIR collections, chosen to span the range of retriever complementarity.
Query counts are those with at least one positive judgement.

| dataset | domain | corpus | train | dev | test | role |
|---|---|---:|---:|---:|---:|---|
| hotpotqa | multi-hop Wikipedia QA | 5.2M | 85,000 | 5,447 | 7,405 | development |
| fever | fact verification | 5.4M | 109,810 | 6,666 | 6,666 | held-out |
| msmarco | web search | 8.8M | 502,939 | 6,980 | 43 | held-out |
| quora | duplicate questions | 523K | — | 5,000 | 10,000 | held-out |
| fiqa | financial QA | 57K | 5,500 | 500 | 648 | held-out |
| nfcorpus | medical IR | 3.6K | 2,590 | 324 | 323 | held-out |
| scifact | scientific claims | 5K | 809 | — | 300 | held-out |

Every held-out dataset needs a non-test split to refit router weights and the
calibration table, which excludes the test-only BEIR collections (arguana,
scidocs, trec-covid, dbpedia-entity, webis-touche2020, climate-fever, nq). Of the
BEIR datasets that do carry a usable fitting split, all are used.

`quora` is a negative control: its oracle-alpha IQR is 0.00, so the study
predicts no gain there.

`src/probe_datasets.py` reports corpus size and per-split query counts for any
candidate dataset before committing to embedding it.

## Split discipline

| split | use |
|---|---|
| train | fit model weights and the calibration table |
| dev | select family, framing, hyperparameters, feature set |
| test | opened once, for the final benchmark |

No decision in this study was made by inspecting test results. `hotpotqa` is the
development dataset and is the only place where design decisions were taken; its
numbers are reported and labelled as such.

## Frozen-spec inheritance

Model and feature selection (pipeline sections 5, 6, 7) run **only** on the
development dataset. Held-out datasets run sections 0-4, 8, 9 and inherit the
selected specification:

| inherited from hotpotqa | refit per dataset |
|---|---|
| model family | model weights |
| label framing | calibration bin-to-alpha table |
| hyperparameters | |
| feature set | |
| decision rule and bin count | |

Nothing is selected on held-out data. The selected specifications are one per
fusion function, so the feature sets that appear across the results table are
three specifications reused, not 21 independent selections.

| fusion | specification | features |
|---|---|---|
| score-minmax | `logreg` / `multibin`, calibrated, 20 bins | `margin_bm25`, `entropy_bm25`, `smv_dense` |
| rrf | `extra_trees` / `multibin`, calibrated, 10 bins | `ql`, `smv_dense`, `d_entropy` |
| borda | `logreg` / `binary`, calibrated, 20 bins | `ql`, `sigma_k_dense`, `d_entropy` |

## Per-dataset overrides

`msmarco` deviates in three respects, declared in `config.yaml` under
`study.overrides`:

| override | value | reason |
|---|---|---|
| `eval_split` | `dev` | BEIR ships msmarco's test split as the 43-query TREC-DL subset, too small to benchmark on; evaluating on dev (6,980 queries) is standard practice |
| `embedding_dtype` | `float16` | 8.8M x 768 is 27 GB at float32, 13.5 GB at float16 |
| `max_fit_queries` | 40,000 | of 502,939 train queries; the router fits on at most `router.train_subset = 10,000` |

Remapping the evaluation split removes `dev` from the fitting and selection
candidates.

## Datasets with only two splits

`scifact` has no dev split, `quora` has no train split, and remapping msmarco's
evaluation split consumes its dev. In these cases the selection split would
otherwise equal the fitting split, and the router would be scored on the queries
it was fitted on.

`load_fit_eval` instead carves a deterministic disjoint half from the available
split (seeded, 50/50). This matters: before the fix, 20 of the 21 configurations
in which a raw-output router significantly beat the constant came from `quora`
and `scifact` alone. A fine-grained raw output can memorise the fitting queries,
whereas quantile binning coarsens and regularises, so the leak inflated the raw
rule specifically.

`sec_final_fit` still fits on the whole split, so its reported dev figure is
in-sample; that is flagged as `dev_in_sample` in `*_router_meta.json`. The
section-9 test evaluation is unaffected in every case.

## Significance

All confidence intervals come from a paired bootstrap over queries, 1000
resamples, seed 42.

```
d = per-query NDCG@10 of method A minus method B      (same queries, aligned)
resample the queries with replacement 1000 times, take the mean of d each time
report the 2.5th and 97.5th percentiles
significant  <=>  the interval excludes zero
```

Pairing removes per-query difficulty, which dominates the variance; two methods
can have widely overlapping per-method intervals while one wins on nearly every
query.

Per-query NDCG@10 for every method and cell is published in
`data/results/router_final/*_benchmark_per_query.csv`, so every interval in the
paper can be recomputed without rerunning the pipeline.

## Statistical caveats

**Non-independence.** The three fusion cells of a dataset share queries,
retrievers and judgements. Treating all 18 held-out cells as independent
overstates the evidence for the complementarity hypothesis. The dataset-level
statistic over 6 independent held-out datasets is reported as the headline;
the cell-level statistic is reported alongside and labelled as
pseudo-replicated.

**Multiple comparisons.** Confidence intervals are per cell, with no family-wise
or false-discovery correction across the 21 cells.

**Statistical versus practical significance.** On `quora`, with 10,000 test
queries, differences of 0.003 NDCG reach significance while being practically
negligible. Effect sizes are reported next to significance flags throughout.

**Power.** `nfcorpus` (323), `scifact` (300) and `fiqa` (648) test queries are
underpowered by construction; small real effects there cannot clear
significance.

## Known limitations

- BM25 hyperparameters were tuned once on `hotpotqa` and inherited by every other
  dataset rather than re-tuned per dataset.
- One dense encoder and one lexical retriever were used; the findings are not
  established for other retriever pairings.
- The calibration bin count was searched over `{10, 20, 50}` jointly with model
  hyperparameters in the same 30-trial TPE study, so it received limited
  dedicated exploration. A one-dimensional sweep would be a stronger test.
- The calibration safety property is an in-sample guarantee; see
  [METHOD.md](METHOD.md) for what holds out of sample.
