"""rescreen_routers.py -- Stage 3: re-screen families x framings x FEATURE SETS.

Stage 1 picked a family on all 46 features; stage 2 pruned to 4 using a LINEAR
workhorse. This stage re-opens both decisions together, because:

  * the best family can change once the feature set shrinks (fewer features
    favour different inductive biases), and
  * the stage-2 ablation is model-specific -- ElasticNet's L1 zeroed a dozen
    features outright (12 consecutive rounds of byte-identical dev NDCG), so it
    found "what the linear router uses", not "what carries signal". A tree may
    exploit features the linear model discarded.

So we screen every (feature_set, family, framing) with an independent Optuna
study, exactly as stage 1 did. Screening several feature sets -- e.g. {3, 4, 11,
46} -- is what prevents the linear model's blind spots from being baked in: if a
tree on 11 features beats a linear on 4, we find out here.

Everything else is identical to stage 1 (same subsample, 80/20 fit/calibrate,
dev scoring, calibrated decision rule, paired bootstrap), so numbers are
directly comparable across stages.

Feature sets are read from the stage-2 ablation path: for each requested size we
take the exact surviving subset at that point on the pruning curve.

Outputs (results/router_screening/):
    <ds>_rescreen.csv              -- every trial
    <ds>_rescreen_best_per_config.csv -- one row per (features, family, framing)
    <ds>_rescreen_best.json        -- the final router: features + family + params
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import optuna

from utils import load_config, get_paths
from screen_routers import (NON_FEATURE, SCALE_SENSITIVE, NO_SAMPLE_WEIGHT, N_THREADS,
                            load_split, ndcg_of_alpha, make_target, _scaler,
                            family_available, build_estimator, run_config,
                            bootstrap_ci, paired_bootstrap, pred_diagnostics)
from ablate_features import feature_cost, COST_CLASS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def feature_sets_from_ablation(paths, name, sizes, all_feats):
    """Pull the exact surviving subset at each requested size off the stage-2 path."""
    p = os.path.join(paths["router_screening"], f"{name}_ablation_path.csv")
    if not os.path.exists(p):
        raise SystemExit(f"[rescreen] {p} missing -- run ablate_features.py first.")
    path = pd.read_csv(p)
    by_n = {int(r["n_features"]): r["features"].split("|") for _, r in path.iterrows()}
    out = {}
    for s in sizes:
        s = int(s)
        if s == len(all_feats):
            out[s] = list(all_feats)
        elif s in by_n:
            out[s] = by_n[s]
        else:
            print(f"[rescreen] no ablation point with {s} features -- skipped.")
    return out


def main():
    config = load_config()
    paths = get_paths(config)
    name = config["dataset"]
    rs = config["router_screen"]
    rc = config.get("rescreen", {})
    seed = int(config.get("seed", 42))
    eval_k = config["retrieval"].get("eval_k", 10)
    n_boot = int(rs.get("bootstrap_resamples", 1000))
    calib_frac = float(rs.get("calib_fraction", 0.2))
    n_jobs = int(rs.get("n_jobs", N_THREADS))
    max_rounds = int(rs.get("max_boost_rounds", 600))
    bins = np.linspace(0.0, 1.0, int(rs.get("n_bins", 11)))
    decision_rules = list(rs.get("decision_rules", ["calibrated"]))
    calib_bin_opts = list(rs.get("n_calib_bins", [10, 20, 50]))
    n_trials = int(rc.get("n_trials", rs.get("n_trials", 30)))
    families = list(rc.get("families", rs["families"]))
    framings = list(rc.get("framings", rs["framings"]))
    sizes = list(rc.get("feature_set_sizes", [3, 4, 11]))

    grid = np.load(os.path.join(paths["feature_dataset"], f"{name}_alpha_grid.npy")).astype(np.float64)
    tr_df, tr_curve_all = load_split(paths, name, "train")
    dv_df, dv_curve = load_split(paths, name, "dev")
    all_feats = [c for c in tr_df.columns if c not in NON_FEATURE]

    fsets = feature_sets_from_ablation(paths, name, sizes, all_feats)
    if not fsets:
        raise SystemExit("[rescreen] no usable feature sets.")

    n_sub = min(int(rs.get("train_subset", 10000)), len(tr_df))
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(tr_df), size=n_sub, replace=False)
    alpha_tr = tr_df.iloc[sub]["alpha"].to_numpy(dtype=np.float64)
    wtr = tr_df.iloc[sub]["alpha_sensitivity"].to_numpy(dtype=np.float64)
    tr_curve_sub = tr_curve_all[sub]
    alpha_dev = dv_df["alpha"].to_numpy(dtype=np.float64)

    n_studies = 0
    for feats in fsets.values():
        for fam in families:
            for fr in framings:
                if family_available(fam, fr):
                    n_studies += 1
    print(f"[rescreen] '{name}': feature sets {sorted(fsets)} x {len(families)} families "
          f"x {len(framings)} framings")
    print(f"[rescreen] {n_studies} studies x {n_trials} trials | threads/job={n_jobs} "
          f"| metric = NDCG@{eval_k}")
    for s, f in sorted(fsets.items()):
        mix = {c: sum(1 for x in f if feature_cost(x) == c) for c in COST_CLASS}
        print(f"[rescreen]   {s:2d} features  cost mix " +
              ", ".join(f"{k}={v}" for k, v in mix.items() if v))

    records, per_query, meta = [], {}, {}

    # ---- reference rows (train-tuned constant + oracle) ----
    a_star = int(np.argmax(tr_curve_sub.mean(axis=0)))
    for label, pq in (("constant_alpha", dv_curve[:, a_star]),
                      ("oracle", dv_curve.max(axis=1))):
        lo, hi = bootstrap_ci(pq, n_boot, seed)
        tag = f"reference|{label}"
        records.append(dict(n_features=0, family="reference", framing=label,
                            params=json.dumps({"alpha_const": float(grid[a_star])}
                                              if label == "constant_alpha" else {}),
                            dev_ndcg=float(pq.mean()), ci_lo=lo, ci_hi=hi, is_best=True))
        per_query[tag] = pq
        meta[tag] = dict(n_features=0, cost=0, pred_std=0.0)
    print(f"[rescreen] constant alpha={grid[a_star]:.2f} -> {per_query['reference|constant_alpha'].mean():.4f}"
          f" | oracle={per_query['reference|oracle'].mean():.4f}\n")

    t0 = time.perf_counter()
    done_studies = 0
    for n_feat in sorted(fsets):
        feats = fsets[n_feat]
        Xtr = tr_df.iloc[sub][feats].to_numpy(dtype=np.float64)
        Xdev = dv_df[feats].to_numpy(dtype=np.float64)
        # worst-case inference cost of this set (drives the tie-break later)
        set_cost = max(COST_CLASS[feature_cost(f)] for f in feats)

        for family in families:
            for framing in framings:
                tag = f"f{n_feat}|{family}|{framing}"
                if not family_available(family, framing):
                    continue
                ytr = make_target(alpha_tr, framing, bins)
                if framing != "regression" and len(np.unique(ytr)) < 2:
                    print(f"[rescreen] {tag}: one class only -- skipped.")
                    continue

                def objective(trial):
                    est = build_estimator(trial, family, framing, seed, n_jobs, max_rounds)
                    if est is None:
                        raise optuna.TrialPruned()
                    use_w = (family not in NO_SAMPLE_WEIGHT and
                             trial.suggest_categorical("use_sample_weight", [True, False]))
                    if family in SCALE_SENSITIVE:
                        est = (_scaler(trial), est)
                    decision = trial.suggest_categorical("decision_rule", decision_rules)
                    n_cb = (trial.suggest_categorical("n_calib_bins", calib_bin_opts)
                            if decision == "calibrated" else 0)
                    try:
                        _, pred, _ = run_config(est, family, framing, use_w, decision, n_cb,
                                                Xtr, ytr, wtr, tr_curve_sub, Xdev, bins,
                                                grid, seed, calib_frac)
                    except Exception:
                        raise optuna.TrialPruned()
                    return float(ndcg_of_alpha(pred, dv_curve, grid).mean())

                t_s = time.perf_counter()
                study = optuna.create_study(direction="maximize",
                                            sampler=optuna.samplers.TPESampler(seed=seed))
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                secs = time.perf_counter() - t_s
                done = [t for t in study.trials if t.value is not None]
                done_studies += 1
                if not done:
                    print(f"[rescreen] [{done_studies}/{n_studies}] {tag}: all pruned ({secs:.0f}s)")
                    continue

                best = study.best_trial
                est = build_estimator(optuna.trial.FixedTrial(best.params), family, framing,
                                      seed, n_jobs, max_rounds)
                use_w = bool(best.params.get("use_sample_weight", False))
                if family in SCALE_SENSITIVE:
                    est = (_scaler(optuna.trial.FixedTrial(best.params)), est)
                decision = best.params.get("decision_rule", "calibrated")
                n_cb = int(best.params.get("n_calib_bins", 0))
                _, pred, _ = run_config(est, family, framing, use_w, decision, n_cb,
                                        Xtr, ytr, wtr, tr_curve_sub, Xdev, bins, grid,
                                        seed, calib_frac)
                pq = ndcg_of_alpha(pred, dv_curve, grid)
                lo, hi = bootstrap_ci(pq, n_boot, seed)
                pm, ps, pc = pred_diagnostics(pred, alpha_dev)

                for t in done:
                    records.append(dict(n_features=n_feat, family=family, framing=framing,
                                        params=json.dumps(t.params), dev_ndcg=float(t.value),
                                        ci_lo=np.nan, ci_hi=np.nan, is_best=False))
                records[-1].update(ci_lo=lo, ci_hi=hi, is_best=True,
                                   params=json.dumps(best.params), dev_ndcg=float(best.value))
                per_query[tag] = pq
                meta[tag] = dict(n_features=n_feat, cost=set_cost, pred_std=ps,
                                 family=family, framing=framing, params=best.params,
                                 features=feats)
                el = time.perf_counter() - t0
                eta = el / done_studies * (n_studies - done_studies)
                print(f"[rescreen] [{done_studies}/{n_studies}] {tag:34s} "
                      f"{best.value:.4f} [{lo:.4f},{hi:.4f}] std={ps:.3f} "
                      f"({secs:.0f}s, ETA {eta/60:.1f}m)")

    # ---- rank; paired bootstrap vs the best MODEL ----
    models = {k: v for k, v in per_query.items() if not k.startswith("reference|")}
    if not models:
        raise SystemExit("[rescreen] no model succeeded.")
    top_tag = max(models, key=lambda k: models[k].mean())
    rows = []
    for tag, pq in per_query.items():
        d, lo, hi = paired_bootstrap(pq, per_query[top_tag], n_boot, seed)
        m = meta[tag]
        rows.append(dict(config=tag, n_features=m["n_features"], max_cost=m.get("cost", 0),
                         dev_ndcg=float(pq.mean()), diff_vs_best=d,
                         diff_ci_lo=lo, diff_ci_hi=hi,
                         tied_with_best=not (lo > 0 or hi < 0),
                         pred_alpha_std=m.get("pred_std", 0.0)))
    cmp_df = pd.DataFrame(rows).sort_values("dev_ndcg", ascending=False).reset_index(drop=True)

    # FINAL PICK: among configs statistically tied with the best, take the
    # cheapest -- fewest features, then lowest inference cost class. With
    # everything inside ~0.003 the nominal maximum is noise; parsimony is the
    # defensible choice and it is what makes the "~1 ms router" claim hold.
    cand = cmp_df[cmp_df["tied_with_best"] & (cmp_df["n_features"] > 0)]
    chosen = cand.sort_values(["n_features", "max_cost", "dev_ndcg"],
                              ascending=[True, True, False]).iloc[0]
    ch = meta[chosen["config"]]

    out_csv = os.path.join(paths["router_screening"], f"{name}_rescreen.csv")
    cmp_csv = os.path.join(paths["router_screening"], f"{name}_rescreen_best_per_config.csv")
    out_json = os.path.join(paths["router_screening"], f"{name}_rescreen_best.json")
    pd.DataFrame(records).to_csv(out_csv, index=False)
    cmp_df.to_csv(cmp_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dict(dataset=name, eval_k=eval_k, chosen_config=chosen["config"],
                       family=ch["family"], framing=ch["framing"], params=ch["params"],
                       n_features=ch["n_features"], features=ch["features"],
                       dev_ndcg=float(chosen["dev_ndcg"]),
                       nominal_best=top_tag,
                       nominal_best_ndcg=float(models[top_tag].mean()),
                       selection_rule="cheapest config (fewest features, then lowest "
                                      "inference cost) among those statistically tied "
                                      "with the nominal best"), f, indent=2)

    print(f"\n[rescreen] wrote {out_csv}\n[rescreen] wrote {cmp_csv}\n[rescreen] wrote {out_json}\n")
    print(cmp_df.head(25).to_string(index=False))
    print(f"\n[rescreen] nominal best : {top_tag} -> {models[top_tag].mean():.4f}")
    print(f"[rescreen] CHOSEN       : {chosen['config']} -> {chosen['dev_ndcg']:.4f} "
          f"({ch['n_features']} features, cheapest among tied)")
    print(f"[rescreen] features     : {', '.join(ch['features'])}")
    print(f"[rescreen] total time   : {(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
