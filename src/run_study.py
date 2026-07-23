"""Run the full (dataset x fusion) study matrix from config.yaml -> `study`.

    nohup python src/run_study.py > study.out 2>&1 &   # run unattended
    python src/run_study.py --dry-run                  # print the matrix only
    python src/run_study.py --status                   # progress so far
    python src/run_study.py --aggregate-only           # rebuild the summary table

Resumable (sections skip when their outputs exist) and fault-tolerant (a failing
cell is logged and the run continues). Per-cell logs go to results/study_logs/.

The development dataset runs the full pipeline (sections 0-9), including model
and feature selection. Held-out datasets run only sections 0-4, 8, 9 and inherit
the development dataset's router spec; only weights and the calibration table are
refit, so nothing is selected on held-out data. Datasets without a train split
fit on dev instead (harmless here since no selection happens).
"""

import os
import sys
import json
import time
import argparse
import traceback
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from utils import load_config, get_paths, dataset_dir
import pipeline as P
import sections as S
from core import read_qrels


def fusion_tag(f):
    return f"score-{f.get('normalizer', 'minmax')}" if f["function"] == "score" else f["function"]


def cell_done(paths, ds, tag):
    return os.path.exists(os.path.join(paths["router_final"], f"{ds}_{tag}_benchmark.csv"))


def build_matrix(cfg):
    st = cfg["study"]
    dev_ds = st["development_dataset"]
    cells = []
    for ds in st["datasets"]:
        for fu in st["fusions"]:
            cells.append(dict(dataset=ds, fusion=fu, tag=fusion_tag(fu),
                              is_dev=(ds == dev_ds)))
    # development dataset first so held-out cells can inherit its spec
    cells.sort(key=lambda c: (not c["is_dev"], c["dataset"], c["tag"]))
    return cells


def run_cell(cell, cfg, paths, log_dir):
    """Run one (dataset, fusion) cell. Returns (status, message)."""
    ds, fu, tag = cell["dataset"], cell["fusion"], cell["tag"]
    st = cfg["study"]

    c = json.loads(json.dumps(cfg))            # deep copy so the caller's cfg is untouched
    c["dataset"] = ds
    c["fusion"] = {**c["fusion"], **fu}
    if not cell["is_dev"]:
        c.setdefault("study", {})["inherit_spec_from"] = st["development_dataset"]

    sec_ids = (st.get("dev_sections", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) if cell["is_dev"]
               else st.get("heldout_sections", [0, 1, 2, 3, 4, 8, 9]))
    SECTIONS = [("download", P.sec_download), ("embed", P.sec_embed),
                ("tune_bm25", P.sec_tune_bm25), ("retrieve", P.sec_retrieve),
                ("dataset", S.sec_dataset), ("screen", S.sec_screen),
                ("ablate", S.sec_ablate), ("rescreen", S.sec_rescreen),
                ("final_fit", S.sec_final_fit), ("benchmark", S.sec_benchmark)]

    log_p = os.path.join(log_dir, f"{ds}_{tag}.log")
    with open(log_p, "a", encoding="utf-8") as log:
        def say(m):
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
            line = f"[{stamp}] {m}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        say(f"=== CELL {ds} / {tag} ({'DEV' if cell['is_dev'] else 'held-out'}) ===")
        for i in sec_ids:
            label, fn = SECTIONS[i]
            t = time.perf_counter()
            try:
                msg = fn(c, paths)
                say(f"  [{i}] {label}: {msg} ({time.perf_counter()-t:.0f}s)")
            except Exception as e:
                say(f"  [{i}] {label}: FAILED -- {type(e).__name__}: {e}")
                log.write(traceback.format_exc() + "\n")
                log.flush()
                return "failed", f"section {i} ({label}): {e}"
        return "done", "ok"


def cell_iqr(paths, ds, tag):
    """Oracle-alpha IQR on the non-test split (dev, else train) -- the H1 x-axis.
    Measuring it off-test means it predicts the test gain rather than being read
    from the same data."""
    _, _, sel = S.resolve_splits(paths, ds)
    p = os.path.join(paths["feature_dataset"], f"{ds}_{tag}_{sel}_features.csv")
    if not os.path.exists(p):
        return np.nan
    a = pd.read_csv(p, usecols=["alpha"])["alpha"].to_numpy()
    q1, q3 = np.percentile(a, [25, 75])
    return float(q3 - q1)


def aggregate(cfg, paths):
    """Collect every finished cell into the study summary table."""
    st = cfg["study"]
    rows = []
    for ds in st["datasets"]:
        for fu in st["fusions"]:
            tag = fusion_tag(fu)
            jp = os.path.join(paths["router_final"], f"{ds}_{tag}_benchmark.json")
            if not os.path.exists(jp):
                continue
            with open(jp, encoding="utf-8") as f:
                b = json.load(f)
            # chosen router spec, for the feature-frequency table across cells
            mp = os.path.join(paths["router_final"], f"{ds}_{tag}_router_meta.json")
            meta = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
            tbl = {r["method"]: r for r in b["table"]}
            base = next((v for k, v in tbl.items() if "[BASELINE]" in k), None)
            router = tbl.get("ROUTER (ours)")
            oracle = tbl.get("Oracle alpha")
            if not (base and router and oracle):
                continue
            head = oracle["ndcg10"] - base["ndcg10"]
            rows.append(dict(
                dataset=ds, fusion=tag, alpha_iqr=round(cell_iqr(paths, ds, tag), 4),
                role="dev" if ds == st["development_dataset"] else "held-out",
                n_queries=b["n_queries"],
                bm25=tbl.get("BM25", {}).get("ndcg10", np.nan),
                dense=tbl.get("Dense", {}).get("ndcg10", np.nan),
                static_best=base["ndcg10"], router=router["ndcg10"],
                oracle=oracle["ndcg10"], headroom=head,
                gain=router["diff_vs_baseline"],
                gain_ci_lo=router["diff_ci_lo"], gain_ci_hi=router["diff_ci_hi"],
                significant=router["significant"],
                pct_headroom=(router["diff_vs_baseline"] / head * 100) if head > 0 else np.nan,
                router_us=b.get("router_us_per_query", np.nan),
                model=f"{meta.get('family','?')}|{meta.get('framing','?')}",
                n_features=len(meta.get("features", [])) or np.nan,
                features="|".join(meta.get("features", []))))
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["alpha_iqr", "dataset", "fusion"],
                                        ascending=[False, True, True])
    out = os.path.join(paths["router_final"], "STUDY_SUMMARY.csv")
    df.to_csv(out, index=False)
    return df, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    paths = get_paths(cfg)
    log_dir = os.path.join(paths["results"], "study_logs")
    os.makedirs(log_dir, exist_ok=True)
    cells = build_matrix(cfg)

    if args.status or args.dry_run:
        print(f"{'dataset':18s} {'fusion':14s} {'role':9s} status")
        for c in cells:
            print(f"{c['dataset']:18s} {c['tag']:14s} "
                  f"{'DEV' if c['is_dev'] else 'held-out':9s} "
                  f"{'DONE' if cell_done(paths, c['dataset'], c['tag']) else 'pending'}")
        n_done = sum(cell_done(paths, c["dataset"], c["tag"]) for c in cells)
        print(f"\n{n_done}/{len(cells)} cells complete")
        if args.dry_run:
            return

    if not args.status and not args.aggregate_only:
        t0 = time.perf_counter()
        print(f"### STUDY: {len(cells)} cells | logs -> {log_dir}", flush=True)
        results = []
        for n, c in enumerate(cells, 1):
            if cell_done(paths, c["dataset"], c["tag"]):
                print(f"[{n}/{len(cells)}] {c['dataset']}/{c['tag']}: already done", flush=True)
                results.append((c, "done", "cached"))
                continue
            print(f"\n[{n}/{len(cells)}] {c['dataset']}/{c['tag']} "
                  f"(elapsed {(time.perf_counter()-t0)/3600:.1f}h)", flush=True)
            status, msg = run_cell(c, cfg, paths, log_dir)
            results.append((c, status, msg))
            if status == "failed":
                print(f"    -> FAILED: {msg} (continuing)", flush=True)
        print(f"\n### finished in {(time.perf_counter()-t0)/3600:.1f}h")
        for c, stt, msg in results:
            if stt == "failed":
                print(f"  FAILED {c['dataset']}/{c['tag']}: {msg}")

    agg = aggregate(cfg, paths)
    if agg is None:
        print("\n[study] no completed cells to aggregate yet.")
        return
    df, out = agg
    pd.set_option("display.width", 220)
    print(f"\n[study] wrote {out}\n")
    print(df[["dataset", "fusion", "role", "alpha_iqr", "static_best", "router",
              "oracle", "headroom", "gain", "significant", "pct_headroom"]].to_string(index=False))
    print("\n[study] chosen router per cell:")
    print(df[["dataset", "fusion", "model", "n_features", "features"]].to_string(index=False))
    # Feature frequency across cells: with tied configs the exact set is
    # interchangeable, so the signal family and count matter more than the names.
    from collections import Counter
    feats = Counter(f for row in df["features"].dropna() for f in row.split("|") if f)
    if feats:
        print(f"\n[study] feature frequency across {len(df)} cells (stability):")
        for f, n in feats.most_common():
            print(f"  {n:2d}/{len(df)}  {f}")
    ok = df[df.role == "held-out"]
    if len(ok) >= 3 and ok["alpha_iqr"].notna().all():
        r = np.corrcoef(ok["alpha_iqr"], ok["gain"])[0, 1]
        print(f"\n[study] H1 check (held-out only): corr(alpha_iqr, gain) = {r:+.3f} "
              f"over {len(ok)} cells")


if __name__ == "__main__":
    main()
