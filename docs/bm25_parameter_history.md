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

## Metric change: NDCG@100 -> NDCG@eval_k (NDCG@10, primary)

`config.yaml` originally used a single `retrieval.top_k` (100) for BOTH the
candidate/Borda pool depth AND the NDCG evaluation cutoff, so **Run 1 and
Run 2 above were both scored with NDCG@100**. `retrieval.top_k` and
`retrieval.eval_k` were later decoupled: `top_k` (100) stays the candidate pool
/ Borda list length N (unchanged, for use in future NDCG@100 evaluation); the
project's **primary metric is now NDCG@eval_k, with `eval_k: 10`**. This
affects `tune_bm25.py`'s scoring metric too (it now optimises NDCG@10, not
NDCG@100). **Run 2's hotpotqa tuning (k1=0.8, b=0.4) was selected under the
old NDCG@100 metric** and has not been re-run under NDCG@10 — the winning
parameters may differ slightly if re-tuned. `alpha_distribution.py`'s output
CSVs now carry an `eval_k` column so future runs are self-documenting; older
CSVs (Run 1, all 14 datasets) lack this column and should be treated as
NDCG@100 per this history file, not silently mixed with NDCG@10 results in
`alpha_summary.csv` (the script now warns if a summary mixes cutoffs).

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
