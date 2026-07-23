"""H2 experiment: raw output vs calibrated decision rule.

    python src/h2_decision_rule.py

For the development dataset and its fusion, screen every (family, framing) under
both decision rules and tabulate them side by side:

  * raw        -- the model output is used directly as the fusion weight alpha.
  * calibrated -- the output ranks queries into bins; each bin emits the alpha
                  maximising its average NDCG curve (histogram binning).

Tests whether a raw router output beats the best single constant alpha (it does
not, across families) and whether calibration recovers a gain. Unlike section 5,
which keeps only the winning rule, this reports both per family.

Output: results/router_final/<ds>_<tag>_h2_decision_rule.csv
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from utils import load_config, get_paths
from core import paired_bootstrap
import sections as S


def fusion_tag(f):
    return f"score-{f.get('normalizer', 'minmax')}" if f["function"] == "score" else f["function"]


def run_cell(cfg, paths, name, fusion):
    """Run the raw-vs-calibrated screen for one (dataset, fusion) cell."""
    cfg = json.loads(json.dumps(cfg))
    cfg["dataset"] = name
    cfg["fusion"] = {**cfg["fusion"], **fusion}
    tag = S.run_tag(cfg)
    out = os.path.join(paths["router_final"], f"{name}_{tag}_h2_decision_rule.csv")
    if os.path.exists(out):
        print(f"### H2  {name} / {tag}: already done -> {out}")
        return pd.read_csv(out), out
    eval_k = cfg["retrieval"].get("eval_k", 10)
    h2 = cfg.get("h2", {})
    families = h2.get("families", cfg["router"]["families"])
    framings = h2.get("framings", cfg["router"]["framings"])
    n_trials = int(h2.get("n_trials", cfg["router"]["n_trials"]))

    c = S._router_ctx(cfg, paths)                       # full feature set
    const_pq, oracle_pq, a_const = S._references(c)
    const = float(const_pq.mean())
    oracle = float(oracle_pq.mean())
    print(f"### H2  {name} / {tag}  | {len(c['feats'])} features | metric NDCG@{eval_k}")
    print(f"reference: constant alpha={a_const:.2f} -> {const:.4f} | oracle -> {oracle:.4f}\n")

    n_boot = int(cfg["router"]["bootstrap_resamples"])
    seed = int(cfg.get("seed", 42))
    rows = [dict(family="reference", framing="constant", rule="-", dev_ndcg=const,
                 pred_std=0.0, beats_constant=False, vs_constant=0.0),
            dict(family="reference", framing="oracle", rule="-", dev_ndcg=oracle,
                 pred_std=0.0, beats_constant=True, vs_constant=oracle - const)]
    per_q = {}                       # (family, framing, rule) -> per-query NDCG

    for fam in families:
        for fr in framings:
            if not S.family_available(fam, fr):
                continue
            for rule in ("raw", "calibrated"):
                c2 = {**c, "rules": [rule]}
                t = time.perf_counter()
                st = S._study(c2, fam, fr, c["Xtr"], c["Xdev"], n_trials)
                if st is None:
                    continue
                _, pred, _ = S._refit_best(c2, fam, fr, st.best_trial.params,
                                           c["Xtr"], c["Xdev"])
                pq = S.ndcg_of_alpha(pred, c["dv_curve"], c["grid"])
                per_q[(fam, fr, rule)] = pq
                pm, ps, _ = S.pred_diag(pred, c["alpha_dev"])
                v = float(pq.mean())
                # paired bootstrap vs the constant baseline over the same queries
                d_, lo, hi = paired_bootstrap(pq, const_pq, n_boot, seed)
                # pred_mean vs a_const tests the mechanism: a model fitted to the
                # oracle-alpha labels targets their mean, which sits well below the
                # alpha that actually maximises NDCG.
                rows.append(dict(family=fam, framing=fr, rule=rule, dev_ndcg=v,
                                 pred_mean=pm, alpha_const=a_const,
                                 pred_std=ps, beats_constant=bool(v > const),
                                 vs_constant=d_, vs_constant_ci_lo=lo,
                                 vs_constant_ci_hi=hi,
                                 sig_vs_constant=bool(lo > 0 or hi < 0)))
                print(f"  {fam}|{fr:11s} {rule:10s} {v:.4f} "
                      f"(vs const {d_:+.4f} CI [{lo:+.4f},{hi:+.4f}]"
                      f"{' SIG' if (lo > 0 or hi < 0) else ''}) std={ps:.3f} "
                      f"({time.perf_counter()-t:.0f}s)")

    # calibrated vs raw, paired over the same queries, per (family, framing)
    for (fam, fr, rule), pq in list(per_q.items()):
        if rule != "calibrated" or (fam, fr, "raw") not in per_q:
            continue
        d_, lo, hi = paired_bootstrap(pq, per_q[(fam, fr, "raw")], n_boot, seed)
        for r in rows:
            if r.get("family") == fam and r.get("framing") == fr and r.get("rule") == "calibrated":
                r["cal_minus_raw"] = d_
                r["cal_vs_raw_ci_lo"] = lo
                r["cal_vs_raw_ci_hi"] = hi
                r["sig_cal_vs_raw"] = bool(lo > 0 or hi < 0)

    df = pd.DataFrame(rows)
    df.insert(0, "fusion", tag)
    df.insert(0, "dataset", name)
    df.to_csv(out, index=False)
    _report(df, const, prefix=f"[{name}/{tag}] ")
    print(f"[h2] wrote {out}\n")
    return df, out


def _report(df, const, prefix=""):
    model = df[df.family != "reference"]
    raw = model[model.rule == "raw"]
    cal = model[model.rule == "calibrated"]
    if not len(raw):
        return
    print(f"\n{prefix}constant baseline: {const:.4f}")
    print(f"{prefix}raw        -> beats constant in {int(raw.beats_constant.sum())}/{len(raw)}"
          f" | mean {raw.dev_ndcg.mean():.4f} | best {raw.dev_ndcg.max():.4f}")
    print(f"{prefix}calibrated -> beats constant in {int(cal.beats_constant.sum())}/{len(cal)}"
          f" | mean {cal.dev_ndcg.mean():.4f} | best {cal.dev_ndcg.max():.4f}")
    if "sig_vs_constant" in model.columns:
        rs = int((raw.sig_vs_constant & (raw.vs_constant < 0)).sum())
        cs = int((cal.sig_vs_constant & (cal.vs_constant > 0)).sum())
        print(f"{prefix}  raw significantly WORSE than constant : {rs}/{len(raw)}")
        print(f"{prefix}  cal significantly BETTER than constant: {cs}/{len(cal)}")
    if "sig_cal_vs_raw" in cal.columns and cal.sig_cal_vs_raw.notna().any():
        print(f"{prefix}  cal significantly beats raw           : "
              f"{int(cal.sig_cal_vs_raw.sum())}/{len(cal)} "
              f"(mean margin {cal.cal_minus_raw.mean():+.4f})")


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    st = cfg.get("study", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None,
                    help="comma-separated; default = the development dataset. "
                         "'all' = every dataset in study.datasets")
    ap.add_argument("--fusions", default=None,
                    help="comma-separated tags (score-minmax,rrf,borda); "
                         "default = the config's fusion. 'all' = study.fusions")
    args = ap.parse_args()

    dev_ds = st.get("development_dataset", cfg["dataset"])
    if args.datasets is None:
        datasets = [dev_ds]
    elif args.datasets == "all":
        datasets = list(st.get("datasets", [dev_ds]))
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_fu = list(st.get("fusions", [cfg["fusion"]]))
    if args.fusions is None:
        fusions = [cfg["fusion"]]
    elif args.fusions == "all":
        fusions = all_fu
    else:
        want = {f.strip() for f in args.fusions.split(",")}
        fusions = [f for f in all_fu if fusion_tag(f) in want]

    print(f"### H2 matrix: {len(datasets)} dataset(s) x {len(fusions)} fusion(s)")
    frames = []
    for ds in datasets:
        for fu in fusions:
            try:
                df, _ = run_cell(cfg, paths, ds, fu)
                frames.append(df)
            except Exception as e:
                print(f"[h2] FAILED {ds}/{fusion_tag(fu)}: {type(e).__name__}: {e}")

    if len(frames) < 2:
        return
    allf = pd.concat(frames, ignore_index=True)
    out = os.path.join(paths["router_final"], "h2_decision_rule_ALL.csv")
    allf.to_csv(out, index=False)
    m = allf[allf.family != "reference"]
    raw, cal = m[m.rule == "raw"], m[m.rule == "calibrated"]
    print("\n" + "=" * 68)
    print(f"[h2] POOLED over {allf.groupby(['dataset','fusion']).ngroups} cells "
          f"-> {out}")
    print(f"  raw        beats constant: {int(raw.beats_constant.sum())}/{len(raw)}")
    print(f"  calibrated beats constant: {int(cal.beats_constant.sum())}/{len(cal)}")
    if "sig_vs_constant" in m.columns:
        print(f"  raw significantly WORSE  : "
              f"{int((raw.sig_vs_constant & (raw.vs_constant < 0)).sum())}/{len(raw)}")
        print(f"  cal significantly BETTER : "
              f"{int((cal.sig_vs_constant & (cal.vs_constant > 0)).sum())}/{len(cal)}")
    # per-cell breakdown, the table to put in the paper
    g = m.groupby(["dataset", "fusion", "rule"]).agg(
        n=("dev_ndcg", "size"), mean_ndcg=("dev_ndcg", "mean"),
        beats=("beats_constant", "sum")).reset_index()
    print("\n" + g.to_string(index=False))


if __name__ == "__main__":
    main()
