# BM25 Parameter History

Record of which BM25 parameters were active for each experiment run, so past
results in `data/results/` can be correctly attributed. **BM25 parameters are
NOT globally fixed** — `src/tune_bm25.py` grid-searches k1/b/stemming per
dataset (see [config.yaml](../config.yaml) `bm25_tuning`), so the `bm25:` block
in config.yaml changes over time as each dataset gets tuned.

## Run 1 — Original / untuned defaults (Lucene / Elasticsearch)

Used for **every** dataset in the current
`data/results/alpha_distribution/` folder — i.e. the full Phase-1 oracle-alpha
benchmark (all `*_alpha.csv`, `*_alpha_boxplot.png`, `alpha_summary.csv`,
`combined_alpha_boxplot.png`) was generated with these parameters, **before**
`tune_bm25.py` existed:

```yaml
bm25:
  method: lucene
  k1: 1.2      # Lucene/Elasticsearch default
  b: 0.75      # Lucene/Elasticsearch default
  use_stemming: true
```

Rationale at the time: no per-corpus tuning existed yet, so we fixed the
classic Lucene/Elasticsearch defaults uniformly across all candidate datasets
(chosen over Anserini's 0.9/0.4, which were tuned for short/uniform MS MARCO
passages — see [config.yaml](../config.yaml) history / prior discussion).

**Datasets covered by this run:** arguana, climate-fever, dbpedia-entity,
fever, fiqa, hotpotqa, msmarco, nfcorpus, nq, quora, scidocs, scifact,
trec-covid, webis-touche2020.

## Run 2 — Per-corpus tuned (hotpotqa)

`src/tune_bm25.py` grid-searched `bm25_tuning.k1 x b x use_stemming` on
**hotpotqa** (mean NDCG@100 over all queries with qrels; full grid in
`data/results/bm25_tuning/`). Result, applied to config.yaml on first tuning:

```yaml
bm25:
  method: lucene
  k1: 0.8      # tuned on hotpotqa
  b: 0.4       # tuned on hotpotqa
  use_stemming: true
```

**Not yet re-run:** the hotpotqa alpha-distribution result currently sitting in
`data/results/alpha_distribution/hotpotqa_alpha.csv` still reflects **Run 1**
parameters (k1=1.2, b=0.75), not this tuned setting.

## Implication for dataset selection

The Phase-1 dataset-selection comparison (`alpha_summary.csv`, the combined
boxplot) is currently **apples-to-apples** — every dataset was scored with the
*same* untuned BM25 (Run 1) — which is valid for comparing datasets against
each other. However, once a specific dataset is chosen for Phase 2, its
alpha-distribution should be **re-run with that dataset's tuned BM25**
(`python src/tune_bm25.py` then `python src/alpha_distribution.py`), since a
better-tuned BM25 shifts the oracle-alpha distribution and NDCG numbers.

## How to extend this log

Each time `tune_bm25.py` is re-run for a new active dataset and its result is
copied into `config.yaml`, add a new dated entry above with: the dataset, the
resulting k1/b/use_stemming, and which `data/results/alpha_distribution/*`
files (if any) still predate that tuning and are therefore due for a re-run.
