# Reproducing the study

## Environment

Linux with an NVIDIA GPU. The study was run on a single RTX 4090 with 24 GB of
VRAM and 32 CPU cores.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CPU-only runs work by dropping the `--index-url` and setting `dense.device: cpu`,
but embedding the larger corpora becomes impractical.

Set `router.n_jobs` to a value below the core count. On a shared machine `-1`
oversubscribes the BLAS thread pool and stalls; 8 is used throughout. Thread caps
are applied in `src/core.py` before numpy is imported and can be overridden with
`PIPE_THREADS`.

## Running everything

```bash
python src/run_study.py
```

This walks the full `dataset x fusion` matrix from `config.yaml`. It is
resumable: every section skips when its outputs already exist, so an interrupted
run continues where it stopped. A cell that fails is logged and the run
continues.

```bash
python src/run_study.py --dry-run          # print the matrix, run nothing
python src/run_study.py --status           # completed cells
python src/run_study.py --aggregate-only   # rebuild STUDY_SUMMARY.csv
```

Long runs should be started inside `tmux` or via `nohup`, since the full matrix
takes days from a cold start.

## The decision-rule experiment

```bash
python src/h2_decision_rule.py --datasets all
```

Screens every (family, framing) pair under both the raw and calibrated decision
rules, with paired-bootstrap intervals, and writes a per-dataset table plus the
pooled `h2_decision_rule_ALL.csv`. Existing per-cell tables are reused, so the
command is safe to re-run.

`--datasets` accepts `all`, a comma-separated list, or is omitted for the
development dataset only. `--fusions` behaves the same way.

## Single dataset

`run_study.py` drives `pipeline.py` per cell; it can also be run directly against
whatever `dataset` and `fusion` are set in `config.yaml`.

```bash
python src/pipeline.py            # all sections
python src/pipeline.py --from 4   # re-run from section 4 onward
python src/pipeline.py --only 9   # one section
```

| # | section | output |
|---|---|---|
| 0 | download | `data/datasets/<ds>/` |
| 1 | embed | `corpus_emb.npy`, memory-mapped |
| 2 | tune_bm25 | `results/bm25_tuning/` (disabled by default) |
| 3 | retrieve | `retrieval_<split>_top1000.npz`, ranked lists and raw scores |
| 4 | dataset | features, alpha-NDCG curves, oracle alpha |
| 5 | screen | families x framings |
| 6 | ablate | greedy backward feature elimination |
| 7 | rescreen | families x framings x feature-set sizes |
| 8 | final_fit | frozen `router.joblib` and `router_meta.json` |
| 9 | benchmark | test evaluation, `benchmark.csv/json`, per-query scores |

Sections 5-7 run only on the development dataset. Section 3 is
fusion-independent and cached, so changing `fusion.function` re-runs section 4
onward but never retrieval.

## Cost

Approximate wall-clock on the reference machine, from a cold start.

| stage | cost |
|---|---|
| embedding a 5M-document corpus | 1-2 h |
| embedding msmarco (8.8M, float16) | 2-4 h |
| retrieval, per split | minutes to ~1 h depending on corpus and query count |
| section 4, per dataset and fusion | minutes |
| screening and ablation (development dataset only) | several hours |
| sections 8-9 per held-out cell | minutes |

Peak disk use is dominated by embeddings and the section-4 curve arrays, roughly
800 GB across all seven datasets if every intermediate is retained.

## Adding a dataset

Check that a candidate has a usable fitting split and an adequately powered
evaluation split before spending time on embeddings:

```bash
python src/probe_datasets.py msmarco nq climate-fever
```

It downloads only, then reports corpus size and per-split query counts with a
verdict. If the evaluation split is too small but another split is large enough,
set `study.overrides.<dataset>.eval_split` as done for msmarco.

## Verifying results without rerunning

The published artifacts are sufficient to check every number in the paper.

| file | contents |
|---|---|
| `router_final/STUDY_SUMMARY.csv` | one row per cell: alpha IQR, baseline, router, oracle, gain, CI, significance |
| `router_final/<ds>_<fusion>_benchmark.csv` | all methods for that cell with NDCG@10/@100, MRR, Recall, CIs |
| `router_final/<ds>_<fusion>_benchmark_per_query.csv` | per-query NDCG@10 for every method |
| `router_final/<ds>_<fusion>_router_meta.json` | the frozen specification and dev-time gain |
| `router_final/h2_decision_rule_ALL.csv` | raw versus calibrated, all datasets, with CIs |
| `router_screening/*_screen.csv` | every family x framing with its tuned hyperparameters |
| `router_screening/*_ablation.csv` | the full backward-elimination path |
| `router_screening/*_rescreen.csv` | families x feature-set sizes, with tie tests |

Confidence intervals can be recomputed directly from the per-query files:

```python
import numpy as np, pandas as pd
d = pd.read_csv("data/results/router_final/fever_score-minmax_benchmark_per_query.csv")
a = d["ROUTER (ours)"].to_numpy()
b = d[[c for c in d.columns if "[BASELINE]" in c][0]].to_numpy()
diff = a - b
rng = np.random.default_rng(42)
means = diff[rng.integers(0, len(diff), size=(1000, len(diff)))].mean(axis=1)
print(diff.mean(), np.percentile(means, [2.5, 97.5]))
```

Intermediate artifacts that are large and fully regenerated (feature tables,
curve arrays, serialised estimators, run logs) are not published; see
`.gitignore`.

## Determinism

`seed: 42` is used for the train subsample, the calibration split, the
fit/evaluation carve on two-split datasets, Optuna sampling, and every bootstrap.
Reruns on the same data reproduce the same numbers. GPU non-determinism in the
encoder can perturb embeddings marginally between hardware generations.
