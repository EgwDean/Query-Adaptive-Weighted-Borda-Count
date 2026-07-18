"""ablate_features.py -- Stage 2: greedy backward feature elimination.

Start from ALL features and prune one at a time:

    round 1: try removing each of the N features -> permanently drop the one
             whose removal hurts LEAST (by dev NDCG@eval_k)
    round 2: repeat on the N-1 survivors
    ...

~N + (N-1) + ... fits total (~1,000 for 46 features), never 2^N combinations.

Why greedy backward rather than group-cutting or permutation importance
----------------------------------------------------------------------
The feature set contains known-redundant families (avgIDF/SCS/AvICTF/gamma1 all
measure specificity; NQC/WIG/SMV/entropy all measure dispersion -- see
docs/ltr_router_features.md). That redundancy breaks the alternatives:
  * group-cutting drops a whole family, taking the one good member with it;
  * permutation importance is worse -- near-duplicate features mask each other,
    so BOTH look useless and both get dropped, losing the signal entirely.
Greedy backward drops one twin, the survivor's contribution immediately rises,
and it is kept. It handles redundancy correctly by construction.

Protocol (identical to stage 1, so numbers stay comparable)
-----------------------------------------------------------
* same seeded 10k train subsample; 80% fit / 20% calibrate; scored on dev
* the same `run_config` fit -> calibrate -> score path as screen_routers.py
* WORKHORSE MODEL: a fast family that was statistically TIED with the stage-1
  winner (elasticnet ~0.6759 vs logreg ~0.6767, paired CI includes 0).
  `logreg` uses the saga solver (~14 s/fit -> ~4 h for the full path);
  elasticnet is sub-second. Stage 3 re-screens ALL families on the surviving
  features, so nothing is lost by pruning with a tied proxy.
* DROP RULE: paired bootstrap of (candidate - full-feature-set) over the same
  dev queries. A feature is only dropped if its removal is NOT significantly
  harmful. Paired (not independent CIs) because it is far more powerful.
* FINAL PICK: parsimony -- the SMALLEST feature set statistically
  indistinguishable from the best point on the path, not the raw argmax
  (which overfits dev).

Caveat: the greedy path makes ~1,000 comparisons against dev and repeatedly
takes a max, so the chosen subset looks slightly better on dev than it truly
is (selection bias). Contained by stage 3 re-screening and by test staying
sealed until stage 5 -- but treat the ablation gain as optimistic.

Outputs (results/router_screening/):
    <ds>_ablation_path.csv      -- one row per pruning round (the NDCG-vs-#features curve)
    <ds>_ablation_rounds.csv    -- every candidate fit (drop order / de-facto importance)
    <ds>_ablation_best.json     -- the selected feature set + parsimony rationale
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Thread caps are applied by screen_routers on import (before numpy) -- see the
# oversubscription note in docs/router_pipeline.md.
import numpy as np
import pandas as pd
import optuna

from utils import load_config, get_paths
from screen_routers import (NON_FEATURE, SCALE_SENSITIVE, NO_SAMPLE_WEIGHT, N_THREADS,
                            load_split, ndcg_of_alpha, make_target, _scaler,
                            build_estimator, run_config, bootstrap_ci,
                            paired_bootstrap, pred_diagnostics)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Inference cost class per feature-name prefix/exact name. Used to break ties in
# favour of the CHEAPER feature set (quality/latency Pareto, not pure accuracy).
#   lookup   = cached term stats / query text only
#   scores   = needs the retrieved top-k SCORES (already in hand)
#   invindex = needs posting-list intersections (inverted index)
#   embed    = needs embedding gathers + a pairwise similarity matrix
#   text     = needs the TEXT of the top-k docs (most expensive)
COST_CLASS = {"lookup": 1, "scores": 2, "invindex": 3, "embed": 4, "text": 5}


def feature_cost(name):
    if name.startswith("clarity"):
        return "text"
    if name.startswith(("autocorr", "apair_ratio")):
        return "embed"
    if name in ("query_scope", "pmi_avg"):
        return "invindex"
    if name.startswith(("top_score", "sigma_k", "margin", "norm_margin", "wig",
                        "nqc", "smv", "entropy", "robust_sigma",
                        "jaccard", "kendall_tau", "d_z", "d_wig")):
        return "scores"
    return "lookup"


def evaluate(feats, ctx):
    """Fit + calibrate on the train subsample, return per-query dev NDCG."""
    idx = [ctx["fidx"][f] for f in feats]
    est = build_estimator(optuna.trial.FixedTrial(ctx["params"]), ctx["family"],
                          ctx["framing"], ctx["seed"], ctx["n_jobs"], ctx["max_rounds"])
    if ctx["family"] in SCALE_SENSITIVE:
        est = (_scaler(optuna.trial.FixedTrial(ctx["params"])), est)
    _, pred, _ = run_config(est, ctx["family"], ctx["framing"], ctx["use_w"],
                            ctx["decision"], ctx["n_cb"],
                            ctx["Xtr"][:, idx], ctx["ytr"], ctx["wtr"], ctx["tr_curve"],
                            ctx["Xdev"][:, idx], ctx["bins"], ctx["grid"],
                            ctx["seed"], ctx["calib_frac"])
    return ndcg_of_alpha(pred, ctx["dv_curve"], ctx["grid"]), pred


def main():
    config = load_config()
    paths = get_paths(config)
    name = config["dataset"]
    rs = config["router_screen"]
    ab = config.get("ablation", {})
    seed = int(config.get("seed", 42))
    eval_k = config["retrieval"].get("eval_k", 10)
    n_boot = int(rs.get("bootstrap_resamples", 1000))
    calib_frac = float(rs.get("calib_fraction", 0.2))
    n_jobs = int(rs.get("n_jobs", N_THREADS))
    max_rounds = int(rs.get("max_boost_rounds", 600))
    bins = np.linspace(0.0, 1.0, int(rs.get("n_bins", 11)))
    min_features = int(ab.get("min_features", 3))

    # ---- workhorse model: from config, else the stage-1 screening winner ----
    best_json = os.path.join(paths["router_screening"], f"{name}_router_screening_best.json")
    family = ab.get("family")
    framing = ab.get("framing")
    params = ab.get("params")
    if family is None or framing is None or params is None:
        if not os.path.exists(best_json):
            raise SystemExit(f"[ablate] {best_json} missing -- run screen_routers.py first "
                             f"(or set ablation.family/framing/params in config.yaml).")
        with open(best_json, encoding="utf-8") as f:
            b = json.load(f)
        family = family or b["best_family"]
        framing = framing or b["best_framing"]
        params = params or b["best_params"]

    grid = np.load(os.path.join(paths["feature_dataset"], f"{name}_alpha_grid.npy")).astype(np.float64)
    tr_df, tr_curve_all = load_split(paths, name, "train")
    dv_df, dv_curve = load_split(paths, name, "dev")

    all_feats = [c for c in tr_df.columns if c not in NON_FEATURE]
    n_sub = min(int(rs.get("train_subset", 10000)), len(tr_df))
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(tr_df), size=n_sub, replace=False)

    ctx = dict(
        family=family, framing=framing, params=params, seed=seed,
        n_jobs=n_jobs, max_rounds=max_rounds, bins=bins, grid=grid,
        calib_frac=calib_frac, dv_curve=dv_curve,
        use_w=bool(params.get("use_sample_weight", False)),
        decision=params.get("decision_rule", "calibrated"),
        n_cb=int(params.get("n_calib_bins", 20) or 20),
        fidx={f: i for i, f in enumerate(all_feats)},
        Xtr=tr_df.iloc[sub][all_feats].to_numpy(dtype=np.float64),
        ytr=make_target(tr_df.iloc[sub]["alpha"].to_numpy(dtype=np.float64), framing, bins),
        wtr=tr_df.iloc[sub]["alpha_sensitivity"].to_numpy(dtype=np.float64),
        tr_curve=tr_curve_all[sub],
        Xdev=dv_df[all_feats].to_numpy(dtype=np.float64),
    )

    n0 = len(all_feats)
    # rounds have sizes n0, n0-1, ..., min_features+1 (each round fits once per
    # surviving feature), plus 1 for the full-set baseline
    total_fits = sum(range(min_features + 1, n0 + 1)) + 1
    print(f"[ablate] '{name}': greedy backward elimination on {n0} features")
    print(f"[ablate] workhorse = {family}|{framing} rule={ctx['decision']}"
          f"[{ctx['n_cb']}] sample_weight={ctx['use_w']}")
    print(f"[ablate] {n_sub:,} train queries (80/20 fit/calibrate) | dev={len(dv_df):,} "
          f"| metric = NDCG@{eval_k}")
    print(f"[ablate] pruning {n0} -> {min_features} features = {total_fits:,} fits total\n")

    t0 = time.perf_counter()
    fit_no = 0

    # ---- baseline: the full feature set ----
    base_pq, base_pred = evaluate(all_feats, ctx)
    fit_no += 1
    base_ndcg = float(base_pq.mean())
    lo, hi = bootstrap_ci(base_pq, n_boot, seed)
    print(f"[ablate] [{fit_no}/{total_fits}] FULL SET ({n0} features): "
          f"dev NDCG@{eval_k}={base_ndcg:.4f} [{lo:.4f}, {hi:.4f}]\n")

    path_rows = [dict(n_features=n0, dropped="-", dropped_cost="-",
                      dev_ndcg=base_ndcg, ci_lo=lo, ci_hi=hi,
                      diff_vs_full=0.0, diff_ci_lo=0.0, diff_ci_hi=0.0,
                      significant_vs_full=False,
                      features="|".join(all_feats))]
    round_rows = []
    current = list(all_feats)
    pq_of = {n0: base_pq}

    # ---- greedy backward elimination ----
    while len(current) > min_features:
        cands = []
        for f in current:
            trial_feats = [c for c in current if c != f]
            pq, _ = evaluate(trial_feats, ctx)
            fit_no += 1
            cands.append((float(pq.mean()), f, pq))
            round_rows.append(dict(n_features=len(current), removed=f,
                                   cost=feature_cost(f), dev_ndcg=float(pq.mean())))
            el = time.perf_counter() - t0
            eta = el / fit_no * (total_fits - fit_no)
            print(f"\r[ablate] [{fit_no}/{total_fits}] round n={len(current)} "
                  f"try drop {f[:28]:28s} -> {pq.mean():.4f} | "
                  f"elapsed {el/60:.1f}m ETA {eta/60:.1f}m", end="", flush=True)

        # keep the removal that hurts least; tie-break toward the COSTLIER feature
        best_score = max(c[0] for c in cands)
        tied = [c for c in cands if best_score - c[0] < 1e-9]
        drop_score, drop_feat, drop_pq = max(
            tied, key=lambda c: COST_CLASS[feature_cost(c[1])])
        current = [c for c in current if c != drop_feat]
        pq_of[len(current)] = drop_pq

        d, dlo, dhi = paired_bootstrap(drop_pq, base_pq, n_boot, seed)
        lo, hi = bootstrap_ci(drop_pq, n_boot, seed)
        sig = bool(dlo > 0 or dhi < 0)
        path_rows.append(dict(n_features=len(current), dropped=drop_feat,
                              dropped_cost=feature_cost(drop_feat),
                              dev_ndcg=drop_score, ci_lo=lo, ci_hi=hi,
                              diff_vs_full=d, diff_ci_lo=dlo, diff_ci_hi=dhi,
                              significant_vs_full=sig,
                              features="|".join(current)))
        print(f"\r[ablate] [{fit_no}/{total_fits}] n={len(current):2d}  dropped "
              f"{drop_feat[:28]:28s} ({feature_cost(drop_feat):8s}) -> {drop_score:.4f} "
              f"[{lo:.4f},{hi:.4f}] vs full {d:+.4f}{' SIG-WORSE' if sig and d < 0 else ''}"
              f"        ")

    path = pd.DataFrame(path_rows)

    # ---- parsimony pick: smallest set statistically tied with the best point ----
    best_row = path.loc[path["dev_ndcg"].idxmax()]
    best_pq = pq_of[int(best_row["n_features"])]
    ok = []
    for _, r in path.iterrows():
        pq = pq_of[int(r["n_features"])]
        _, plo, phi = paired_bootstrap(pq, best_pq, n_boot, seed)
        tied_with_best = not (plo > 0 or phi < 0)          # CI of the diff includes 0
        ok.append(tied_with_best)
    path["tied_with_best"] = ok
    chosen = path[path["tied_with_best"]].sort_values("n_features").iloc[0]
    chosen_feats = chosen["features"].split("|")

    out_path = os.path.join(paths["router_screening"], f"{name}_ablation_path.csv")
    out_rounds = os.path.join(paths["router_screening"], f"{name}_ablation_rounds.csv")
    out_json = os.path.join(paths["router_screening"], f"{name}_ablation_best.json")
    path.to_csv(out_path, index=False)
    pd.DataFrame(round_rows).to_csv(out_rounds, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dict(dataset=name, eval_k=eval_k, workhorse=f"{family}|{framing}",
                       params=params, n_features_full=n0, full_ndcg=base_ndcg,
                       best_ndcg=float(best_row["dev_ndcg"]),
                       best_n_features=int(best_row["n_features"]),
                       chosen_n_features=int(chosen["n_features"]),
                       chosen_ndcg=float(chosen["dev_ndcg"]),
                       chosen_features=chosen_feats,
                       cost_mix={c: sum(1 for f in chosen_feats if feature_cost(f) == c)
                                 for c in COST_CLASS},
                       selection_rule="smallest feature set whose paired-bootstrap CI "
                                      "vs the best path point includes 0 (parsimony)"),
                  f, indent=2)

    print(f"\n[ablate] wrote {out_path}\n[ablate] wrote {out_rounds}\n[ablate] wrote {out_json}\n")
    cols = ["n_features", "dropped", "dropped_cost", "dev_ndcg", "diff_vs_full",
            "significant_vs_full", "tied_with_best"]
    print(path[cols].to_string(index=False))
    print(f"\n[ablate] full set   : {n0} features -> {base_ndcg:.4f}")
    print(f"[ablate] best point : {int(best_row['n_features'])} features -> {best_row['dev_ndcg']:.4f}")
    print(f"[ablate] CHOSEN     : {int(chosen['n_features'])} features -> {chosen['dev_ndcg']:.4f} "
          f"(smallest set tied with best)")
    print(f"[ablate] cost mix   : " +
          ", ".join(f"{c}={sum(1 for f in chosen_feats if feature_cost(f) == c)}"
                    for c in COST_CLASS))
    print(f"[ablate] features   : {', '.join(chosen_feats)}")
    print(f"[ablate] total time : {(time.perf_counter() - t0)/60:.1f} min")
    print("\n[ablate] NOTE: the greedy path takes ~1000 maxima against dev, so this "
          "gain is optimistic (selection bias). Stage 3 re-screens all families on "
          "these features; test stays sealed until stage 5.")


if __name__ == "__main__":
    main()
