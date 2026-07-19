"""final_fit.py -- Stage 4: fit the chosen router on the FULL train split and freeze it.

Takes the configuration selected by stage 3 (`<ds>_rescreen_best.json`: feature
set + family + framing + hyperparameters + decision rule) and refits it on the
entire train split instead of the 10k screening subsample.

Why this is a separate script from the benchmark
------------------------------------------------
This stage FREEZES the router. Stage 5 then opens the test split exactly once
against a fixed artefact. Keeping them apart is what stops "evaluate on test ->
tweak -> re-evaluate", which would invalidate the whole result. Nothing here
touches test, and nothing here is selected against dev -- every choice was
already made in stages 1-3; dev is scored only as a consistency check.

The one thing that genuinely improves at full scale is the CALIBRATION TABLE:
the 20% held-out slice grows from ~2,000 queries (~100/bin) to ~17,000
(~1,700/bin), so each bin's average alpha->NDCG curve is far better estimated.

Outputs (results/router_final/):
    <ds>_router.joblib     -- fitted pipeline + calibration table + feature list
    <ds>_router_meta.json  -- human-readable spec and the reference scores
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib
import optuna

from utils import load_config, get_paths
from screen_routers import (NON_FEATURE, SCALE_SENSITIVE, N_THREADS, load_split,
                            ndcg_of_alpha, make_target, _scaler, build_estimator,
                            run_config, bootstrap_ci, paired_bootstrap,
                            pred_diagnostics, predict_alpha, apply_calibration)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    config = load_config()
    paths = get_paths(config)
    name = config["dataset"]
    rs = config["router_screen"]
    seed = int(config.get("seed", 42))
    eval_k = config["retrieval"].get("eval_k", 10)
    n_boot = int(rs.get("bootstrap_resamples", 1000))
    calib_frac = float(rs.get("calib_fraction", 0.2))
    n_jobs = int(rs.get("n_jobs", N_THREADS))
    max_rounds = int(rs.get("max_boost_rounds", 600))
    bins = np.linspace(0.0, 1.0, int(rs.get("n_bins", 11)))

    spec_path = os.path.join(paths["router_screening"], f"{name}_rescreen_best.json")
    if not os.path.exists(spec_path):
        raise SystemExit(f"[final] {spec_path} missing -- run rescreen_routers.py first.")
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)

    family, framing = spec["family"], spec["framing"]
    params = spec["params"]
    feats = spec["features"]
    decision = params.get("decision_rule", "calibrated")
    n_cb = int(params.get("n_calib_bins", 20) or 20)
    use_w = bool(params.get("use_sample_weight", False))

    grid = np.load(os.path.join(paths["feature_dataset"], f"{name}_alpha_grid.npy")).astype(np.float64)
    tr_df, tr_curve = load_split(paths, name, "train")
    dv_df, dv_curve = load_split(paths, name, "dev")

    missing = [f for f in feats if f not in tr_df.columns]
    if missing:
        raise SystemExit(f"[final] features missing from the dataset: {missing}")

    Xtr = tr_df[feats].to_numpy(dtype=np.float64)
    alpha_tr = tr_df["alpha"].to_numpy(dtype=np.float64)
    ytr = make_target(alpha_tr, framing, bins)
    wtr = tr_df["alpha_sensitivity"].to_numpy(dtype=np.float64)
    Xdev = dv_df[feats].to_numpy(dtype=np.float64)
    alpha_dev = dv_df["alpha"].to_numpy(dtype=np.float64)

    n_cal = int(calib_frac * len(Xtr))
    print(f"[final] '{name}': fitting {family}|{framing} on the FULL train split")
    print(f"[final] features ({len(feats)}): {', '.join(feats)}")
    print(f"[final] rule={decision}[{n_cb}] sample_weight={use_w} | params={params}")
    print(f"[final] train={len(Xtr):,} -> ~{len(Xtr)-n_cal:,} fit / ~{n_cal:,} calibrate "
          f"(~{n_cal//max(n_cb,1):,} queries per bin, vs ~{2000//max(n_cb,1)} at screening)")

    t0 = time.perf_counter()
    est = build_estimator(optuna.trial.FixedTrial(params), family, framing, seed,
                          n_jobs, max_rounds)
    if est is None:
        raise SystemExit(f"[final] cannot build {family}|{framing} with these params.")
    if family in SCALE_SENSITIVE:
        est = (_scaler(optuna.trial.FixedTrial(params)), est)

    model, dev_pred, calib = run_config(est, family, framing, use_w, decision, n_cb,
                                        Xtr, ytr, wtr, tr_curve, Xdev, bins, grid,
                                        seed, calib_frac)
    fit_secs = time.perf_counter() - t0

    # ---- reference scores on dev (consistency check only -- NOT selection) ----
    pq = ndcg_of_alpha(dev_pred, dv_curve, grid)
    lo, hi = bootstrap_ci(pq, n_boot, seed)
    pm, ps, pc = pred_diagnostics(dev_pred, alpha_dev)
    a_star = int(np.argmax(tr_curve.mean(axis=0)))          # constant tuned on FULL train
    const_pq = dv_curve[:, a_star]
    oracle_pq = dv_curve.max(axis=1)
    d, dlo, dhi = paired_bootstrap(pq, const_pq, n_boot, seed)

    print(f"\n[final] fitted in {fit_secs:.1f}s")
    print(f"[final] dev NDCG@{eval_k} = {pq.mean():.4f} [{lo:.4f}, {hi:.4f}]  "
          f"(screening dev was {spec.get('dev_ndcg', float('nan')):.4f})")
    print(f"[final] constant alpha={grid[a_star]:.2f} -> {const_pq.mean():.4f} | "
          f"oracle -> {oracle_pq.mean():.4f}")
    print(f"[final] vs constant: {d:+.4f} CI [{dlo:+.4f}, {dhi:+.4f}] "
          f"{'SIGNIFICANT' if (dlo > 0 or dhi < 0) else 'not significant'}")
    print(f"[final] predicted alpha: mean={pm:.3f} std={ps:.3f} "
          f"({'DEGENERATE' if ps < 0.01 else 'varies -> routing'})")
    if calib is not None:
        edges, bin_alpha = calib
        print(f"[final] calibration table ({len(bin_alpha)} bins): "
              f"alpha {bin_alpha.min():.2f}..{bin_alpha.max():.2f}, "
              f"{len(np.unique(bin_alpha))} distinct values")

    # ---- freeze ----
    art = os.path.join(paths["router_final"], f"{name}_router.joblib")
    joblib.dump(dict(model=model, family=family, framing=framing, features=feats,
                     bins=bins, decision=decision,
                     calib_edges=(calib[0] if calib else None),
                     calib_bin_alpha=(calib[1] if calib else None),
                     alpha_grid=grid), art)
    meta = os.path.join(paths["router_final"], f"{name}_router_meta.json")
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(dict(dataset=name, eval_k=eval_k, family=family, framing=framing,
                       features=feats, n_features=len(feats), params=params,
                       decision_rule=decision, n_calib_bins=n_cb,
                       use_sample_weight=use_w,
                       n_train=len(Xtr), n_calibrate=n_cal, fit_seconds=fit_secs,
                       dev_ndcg=float(pq.mean()), dev_ci=[lo, hi],
                       dev_constant_ndcg=float(const_pq.mean()),
                       dev_constant_alpha=float(grid[a_star]),
                       dev_oracle_ndcg=float(oracle_pq.mean()),
                       dev_gain_vs_constant=d, dev_gain_ci=[dlo, dhi],
                       pred_alpha_mean=pm, pred_alpha_std=ps,
                       note="FROZEN. Stage 5 opens test exactly once against this "
                            "artefact. Dev numbers here are a consistency check, "
                            "not a selection criterion."), f, indent=2)

    print(f"\n[final] wrote {art}\n[final] wrote {meta}")
    print("[final] router FROZEN -- stage 5 evaluates it on test exactly once.")


if __name__ == "__main__":
    main()
