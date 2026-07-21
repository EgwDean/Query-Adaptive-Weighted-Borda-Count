"""h2_decision_rule.py -- the H2 experiment: raw output vs calibrated decision rule.

    python src/h2_decision_rule.py

For the development dataset + its fusion (config `dataset` / `fusion`), screen
every (family, framing) under BOTH decision rules and tabulate them side by side:

  * raw        -- the model's output is used directly as the fusion weight alpha.
  * calibrated -- the output only RANKS queries into bins; each bin emits the
                  alpha maximising its average NDCG curve (histogram binning).

H2 claim: a trained router's raw output is a proxy/probability, NOT a fusion
weight, so used directly it LOSES to the best single constant alpha -- across
model families. Histogram-binned calibration fixes this and, by construction,
cannot do worse than the constant. This script produces the evidence table.

Unlike section 5 (which searches both rules and keeps the winner), this keeps
BOTH outcomes per family, which is the whole point.

Output: results/router_final/<ds>_<tag>_h2_decision_rule.csv
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from utils import load_config, get_paths
import sections as S


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    cfg["dataset"] = cfg.get("study", {}).get("development_dataset", cfg["dataset"])
    name = cfg["dataset"]
    tag = S.run_tag(cfg)
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

    rows = [dict(family="reference", framing="constant", rule="-", dev_ndcg=const,
                 pred_std=0.0, beats_constant=False, vs_constant=0.0),
            dict(family="reference", framing="oracle", rule="-", dev_ndcg=oracle,
                 pred_std=0.0, beats_constant=True, vs_constant=oracle - const)]

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
                _, ps, _ = S.pred_diag(pred, c["alpha_dev"])
                v = float(pq.mean())
                rows.append(dict(family=fam, framing=fr, rule=rule, dev_ndcg=v,
                                 pred_std=ps, beats_constant=bool(v > const),
                                 vs_constant=v - const))
                print(f"  {fam}|{fr:11s} {rule:10s} {v:.4f} "
                      f"(vs const {v - const:+.4f}) std={ps:.3f} "
                      f"({time.perf_counter()-t:.0f}s)")

    df = pd.DataFrame(rows)
    out = os.path.join(paths["router_final"], f"{name}_{tag}_h2_decision_rule.csv")
    df.to_csv(out, index=False)

    # summary: how the two rules compare across families
    model = df[df.family != "reference"]
    raw = model[model.rule == "raw"]
    cal = model[model.rule == "calibrated"]
    print(f"\n[h2] wrote {out}\n")
    print(f"constant baseline           : {const:.4f}")
    print(f"raw rule       -> beats constant in {int(raw.beats_constant.sum())}/{len(raw)} "
          f"configs | mean {raw.dev_ndcg.mean():.4f} | best {raw.dev_ndcg.max():.4f}")
    print(f"calibrated rule-> beats constant in {int(cal.beats_constant.sum())}/{len(cal)} "
          f"configs | mean {cal.dev_ndcg.mean():.4f} | best {cal.dev_ndcg.max():.4f}")
    if len(raw) and raw.dev_ndcg.max() < const:
        print("[h2] CONFIRMED: NOT ONE raw-output router beats the constant baseline.")
    print("[h2] The gap between the two rules is the decision-layer contribution.")


if __name__ == "__main__":
    main()
