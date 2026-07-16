"""screen_routers.py -- Stage 1: compare model families x framings for the alpha router.

For every (family, framing) pair, run an INDEPENDENT Bayesian-optimisation study
(Optuna/TPE) over that family's hyperparameters: fit on a fixed subset of the
train split, then score the trial by the metric that actually matters --
mean NDCG@eval_k on dev, obtained by looking the predicted alpha up in the
precomputed alpha->NDCG curve from create_dataset.py.

Design notes
------------
* Per-family studies (not one BO over a mixed/conditional space) so every family
  gets an identical trial budget and the comparison is best-vs-best.
* Selection metric is ALWAYS NDCG@eval_k. MSE/CE are recorded as diagnostics
  only: the alpha->NDCG curve is flat over wide plateaus, so a large alpha error
  can cost zero NDCG (and a small one can cost a lot).
* Scaling is a searched hyperparameter (standard/robust/quantile) for
  scale-sensitive families; tree ensembles get none. Scalers live INSIDE the
  pipeline, so they are fit on train only -- never on dev.
* `alpha_sensitivity` (max-min NDCG over the alpha grid) is offered as a sample
  weight: queries whose curve is flat cannot affect the end metric whatever we
  predict, so they should not consume model capacity. Searched on/off.
* Reference rows (constant-alpha, oracle) are included for interpretability --
  they are NOT the formal benchmark, which happens later against all baselines.

Outputs (results/router_screening/):
    <dataset>_router_screening.csv       -- every trial, ranked by dev NDCG
    <dataset>_router_screening_best.json -- best overall + best per (family, framing)
"""

import os
import sys
import json
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              ExtraTreesRegressor, ExtraTreesClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, log_loss

from utils import load_config, get_paths

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# Optional gradient-boosting libs: skip the family with a warning if absent.
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except ImportError:
    LGBMRegressor = LGBMClassifier = None
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = XGBClassifier = None
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError:
    CatBoostRegressor = CatBoostClassifier = None

# Columns that are identifiers / labels / references -- never model inputs.
NON_FEATURE = {
    "dataset", "split", "qid",
    "alpha", "oracle_ndcg", "bm25_ndcg", "dense_ndcg",
    "alpha_sensitivity", "plateau_frac", "n_rel", "eval_k", "top_k",
}
# Families that need feature scaling (trees do not).
SCALE_SENSITIVE = {"elasticnet", "logreg", "mlp"}
# sklearn's MLP does not accept sample_weight in fit().
NO_SAMPLE_WEIGHT = {"mlp"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split(paths, name, split):
    fd = paths["feature_dataset"]
    csv = os.path.join(fd, f"{name}_{split}_features.csv")
    npy = os.path.join(fd, f"{name}_{split}_alpha_ndcg_curve.npy")
    for p in (csv, npy):
        if not os.path.exists(p):
            raise SystemExit(f"[screen] missing {p} -- run create_dataset.py first.")
    return pd.read_csv(csv), np.load(npy)


def ndcg_of_alpha(pred_alpha, curve, grid):
    """Per-query NDCG@eval_k for a predicted alpha: nearest grid point -> lookup."""
    pred = np.clip(np.asarray(pred_alpha, dtype=np.float64), 0.0, 1.0)
    idx = np.abs(grid[None, :] - pred[:, None]).argmin(axis=1)
    return curve[np.arange(curve.shape[0]), idx]


# --------------------------------------------------------------------------- #
# Framings: alpha <-> model target
# --------------------------------------------------------------------------- #
def make_target(alpha, framing, bins):
    if framing == "regression":
        return alpha.astype(np.float64)
    if framing == "binary":
        return (alpha > 0.5).astype(int)
    if framing == "multibin":
        return np.abs(alpha[:, None] - bins[None, :]).argmin(axis=1)
    raise ValueError(framing)


def predict_alpha(est, X, framing, bins):
    if framing == "regression":
        return np.clip(est.predict(X), 0.0, 1.0)
    proba = est.predict_proba(X)
    classes = np.asarray(est.classes_)
    if framing == "binary":
        if 1 not in classes:            # degenerate: no positive examples seen
            return np.zeros(X.shape[0])
        return proba[:, list(classes).index(1)]
    return proba @ bins[classes]        # multibin: expected value over bin centres


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #
def _scaler(trial):
    kind = trial.suggest_categorical("scaler", ["standard", "robust", "quantile"])
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    return QuantileTransformer(output_distribution="normal", subsample=100000,
                               random_state=0)


def family_available(family, framing):
    """Is this (family, framing) runnable? Checked WITHOUT building a trial --
    probing with a FixedTrial would raise on the first missing suggestion."""
    is_reg = framing == "regression"
    if family == "lightgbm" and LGBMRegressor is None:
        return False
    if family == "xgboost" and XGBRegressor is None:
        return False
    if family == "catboost" and CatBoostRegressor is None:
        return False
    if family == "elasticnet" and not is_reg:      # regression only
        return False
    if family == "logreg" and is_reg:              # classification only
        return False
    return True


def build_estimator(trial, family, framing, seed):
    """Return an estimator for this trial, or None if the family/framing is unavailable."""
    is_reg = framing == "regression"

    if family == "lightgbm":
        if LGBMRegressor is None:
            return None
        p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 1500, log=True),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 subsample_freq=1,
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                 n_jobs=-1, random_state=seed, verbose=-1)
        return LGBMRegressor(**p) if is_reg else LGBMClassifier(**p)

    if family == "xgboost":
        if XGBRegressor is None:
            return None
        p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 1500, log=True),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 max_depth=trial.suggest_int("max_depth", 3, 12),
                 min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                 n_jobs=-1, random_state=seed, tree_method="hist", verbosity=0)
        return XGBRegressor(**p) if is_reg else XGBClassifier(**p)

    if family == "catboost":
        if CatBoostRegressor is None:
            return None
        p = dict(iterations=trial.suggest_int("iterations", 100, 1500, log=True),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 depth=trial.suggest_int("depth", 4, 10),
                 l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                 random_seed=seed, verbose=0, allow_writing_files=False)
        return CatBoostRegressor(**p) if is_reg else CatBoostClassifier(**p)

    if family == "hist_gbdt":
        p = dict(max_iter=trial.suggest_int("max_iter", 100, 1000, log=True),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 100),
                 l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
                 random_state=seed)
        return HistGradientBoostingRegressor(**p) if is_reg else HistGradientBoostingClassifier(**p)

    if family in ("random_forest", "extra_trees"):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 800, log=True),
                 max_depth=trial.suggest_int("max_depth", 5, 30),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                 max_features=trial.suggest_float("max_features", 0.2, 1.0),
                 n_jobs=-1, random_state=seed)
        if family == "random_forest":
            return RandomForestRegressor(**p) if is_reg else RandomForestClassifier(**p)
        return ExtraTreesRegressor(**p) if is_reg else ExtraTreesClassifier(**p)

    if family == "elasticnet":                       # regression only
        if not is_reg:
            return None
        return ElasticNet(alpha=trial.suggest_float("alpha", 1e-5, 10.0, log=True),
                          l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
                          max_iter=5000, random_state=seed)

    if family == "logreg":                           # classification only
        if is_reg:
            return None
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
        p = dict(C=trial.suggest_float("C", 1e-4, 100.0, log=True),
                 penalty=penalty, solver="saga", max_iter=3000, n_jobs=-1,
                 random_state=seed)
        if penalty == "elasticnet":
            p["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        return LogisticRegression(**p)

    if family == "mlp":
        width = trial.suggest_categorical("width", [32, 64, 128, 256])
        depth = trial.suggest_int("depth", 1, 3)
        p = dict(hidden_layer_sizes=tuple([width] * depth),
                 alpha=trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
                 learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                 max_iter=trial.suggest_int("max_iter", 200, 600),
                 early_stopping=True, n_iter_no_change=15, random_state=seed)
        return MLPRegressor(**p) if is_reg else MLPClassifier(**p)

    return None


# --------------------------------------------------------------------------- #
# Fit / evaluate one configuration
# --------------------------------------------------------------------------- #
def fit_predict(est, family, use_w, Xtr, ytr, wtr, Xdev, framing, bins):
    if family in SCALE_SENSITIVE:
        pipe = Pipeline([("scaler", est[0]), ("model", est[1])])
        if use_w:
            pipe.fit(Xtr, ytr, model__sample_weight=wtr)
        else:
            pipe.fit(Xtr, ytr)
        model = pipe
    else:
        model = est
        if use_w:
            model.fit(Xtr, ytr, sample_weight=wtr)
        else:
            model.fit(Xtr, ytr)
    return model, predict_alpha(model, Xdev, framing, bins)


def bootstrap_ci(x, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a, b, n, seed):
    """Paired bootstrap of (a-b) over queries: mean diff + 95% CI of the diff."""
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return float(d.mean()), lo, hi


# --------------------------------------------------------------------------- #
def main():
    config = load_config()
    paths = get_paths(config)
    name = config["dataset"]
    rs = config["router_screen"]
    seed = int(config.get("seed", 42))
    eval_k = config["retrieval"].get("eval_k", 10)
    bins = np.linspace(0.0, 1.0, int(rs.get("n_bins", 11)))
    n_boot = int(rs.get("bootstrap_resamples", 1000))

    grid = np.load(os.path.join(paths["feature_dataset"], f"{name}_alpha_grid.npy")).astype(np.float64)
    tr_df, tr_curve = load_split(paths, name, "train")
    dv_df, dv_curve = load_split(paths, name, "dev")

    feats = [c for c in tr_df.columns if c not in NON_FEATURE]
    print(f"[screen] '{name}': {len(feats)} features | train={len(tr_df):,} dev={len(dv_df):,} "
          f"| metric = NDCG@{eval_k}")

    # fixed, seeded subsample of train
    n_sub = min(int(rs.get("train_subset", 10000)), len(tr_df))
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(tr_df), size=n_sub, replace=False)
    Xtr = tr_df.iloc[sub][feats].to_numpy(dtype=np.float64)
    alpha_tr = tr_df.iloc[sub]["alpha"].to_numpy(dtype=np.float64)
    wtr = tr_df.iloc[sub]["alpha_sensitivity"].to_numpy(dtype=np.float64)
    tr_curve_sub = tr_curve[sub]
    Xdev = dv_df[feats].to_numpy(dtype=np.float64)
    alpha_dev = dv_df["alpha"].to_numpy(dtype=np.float64)
    print(f"[screen] screening on {n_sub:,} train queries")

    records, per_query = [], {}

    # ---- reference rows (free from the curve; NOT the formal benchmark) ----
    a_star_idx = int(np.argmax(tr_curve_sub.mean(axis=0)))          # tuned on TRAIN only
    const_pq = dv_curve[:, a_star_idx]
    oracle_pq = dv_curve.max(axis=1)
    for label, pq, extra in (("constant_alpha", const_pq, {"alpha_const": float(grid[a_star_idx])}),
                             ("oracle", oracle_pq, {})):
        lo, hi = bootstrap_ci(pq, n_boot, seed)
        records.append(dict(family="reference", framing=label, params=json.dumps(extra),
                            dev_ndcg=float(pq.mean()), ci_lo=lo, ci_hi=hi,
                            dev_mse=np.nan, dev_ce=np.nan, is_best=False))
        per_query[f"reference|{label}"] = pq
    print(f"[screen] reference: constant alpha={grid[a_star_idx]:.2f} -> dev NDCG@{eval_k}="
          f"{const_pq.mean():.4f} | oracle={oracle_pq.mean():.4f}")

    # ---- one Optuna study per (family, framing) ----
    n_trials = int(rs.get("n_trials", 50))
    for family in rs["families"]:
        for framing in rs["framings"]:
            tag = f"{family}|{framing}"
            if not family_available(family, framing):
                print(f"[screen] {tag}: unavailable (missing library or invalid framing) -- skipped.")
                continue

            ytr = make_target(alpha_tr, framing, bins)
            if framing != "regression" and len(np.unique(ytr)) < 2:
                print(f"[screen] {tag}: only one class in the subsample -- skipped.")
                continue

            def objective(trial):
                est = build_estimator(trial, family, framing, seed)
                if est is None:
                    raise optuna.TrialPruned()
                use_w = (family not in NO_SAMPLE_WEIGHT and
                         trial.suggest_categorical("use_sample_weight", [True, False]))
                if family in SCALE_SENSITIVE:
                    est = (_scaler(trial), est)
                try:
                    _, pred = fit_predict(est, family, use_w, Xtr, ytr, wtr, Xdev, framing, bins)
                except Exception:
                    raise optuna.TrialPruned()
                return float(ndcg_of_alpha(pred, dv_curve, grid).mean())

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            done = [t for t in study.trials if t.value is not None]
            if not done:
                print(f"[screen] {tag}: all trials pruned -- skipped.")
                continue

            # refit the best trial to capture per-query dev NDCG + diagnostics
            best = study.best_trial
            est = build_estimator(optuna.trial.FixedTrial(best.params), family, framing, seed)
            use_w = bool(best.params.get("use_sample_weight", False))
            if family in SCALE_SENSITIVE:
                est = (_scaler(optuna.trial.FixedTrial(best.params)), est)
            model, pred = fit_predict(est, family, use_w, Xtr, ytr, wtr, Xdev, framing, bins)
            pq = ndcg_of_alpha(pred, dv_curve, grid)
            lo, hi = bootstrap_ci(pq, n_boot, seed)

            # diagnostics only -- never used for selection
            mse = float(mean_squared_error(alpha_dev, np.clip(pred, 0, 1)))
            ce = np.nan
            if framing != "regression":
                try:
                    ce = float(log_loss(make_target(alpha_dev, framing, bins),
                                        model.predict_proba(Xdev), labels=model.classes_))
                except Exception:
                    ce = np.nan

            # log EVERY trial; then enrich the row belonging to the best trial
            # (matched by trial number -- the best is not necessarily the last).
            best_row = None
            for t in done:
                records.append(dict(family=family, framing=framing,
                                    params=json.dumps(t.params), dev_ndcg=float(t.value),
                                    ci_lo=np.nan, ci_hi=np.nan, dev_mse=np.nan, dev_ce=np.nan,
                                    is_best=False))
                if t.number == best.number:
                    best_row = records[-1]
            if best_row is not None:
                best_row.update(ci_lo=lo, ci_hi=hi, dev_mse=mse, dev_ce=ce, is_best=True)
            per_query[tag] = pq
            print(f"[screen] {tag:28s} best dev NDCG@{eval_k}={best.value:.4f} "
                  f"[{lo:.4f}, {hi:.4f}]  ({len(done)} trials)")

    # ---- rank + paired bootstrap vs the best MODEL (references excluded) ----
    df = pd.DataFrame(records).sort_values("dev_ndcg", ascending=False).reset_index(drop=True)
    models = df[(df.family != "reference") & df.ci_lo.notna()]
    if models.empty:
        raise SystemExit("[screen] no model succeeded.")
    top = models.iloc[0]
    top_tag = f"{top.family}|{top.framing}"
    rows = []
    for tag, pq in per_query.items():
        d, lo, hi = paired_bootstrap(pq, per_query[top_tag], n_boot, seed)
        rows.append(dict(config=tag, dev_ndcg=float(pq.mean()), diff_vs_best=d,
                         diff_ci_lo=lo, diff_ci_hi=hi,
                         significant=bool(lo > 0 or hi < 0)))
    cmp_df = pd.DataFrame(rows).sort_values("dev_ndcg", ascending=False).reset_index(drop=True)

    out_csv = os.path.join(paths["router_screening"], f"{name}_router_screening.csv")
    cmp_csv = os.path.join(paths["router_screening"], f"{name}_router_screening_best_per_config.csv")
    out_json = os.path.join(paths["router_screening"], f"{name}_router_screening_best.json")
    df.to_csv(out_csv, index=False)
    cmp_df.to_csv(cmp_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dict(dataset=name, eval_k=eval_k, n_train_subset=n_sub,
                       n_features=len(feats), features=feats,
                       best_family=top.family, best_framing=top.framing,
                       best_params=json.loads(top.params), best_dev_ndcg=float(top.dev_ndcg)),
                  f, indent=2)

    print(f"\n[screen] wrote {out_csv}\n[screen] wrote {cmp_csv}\n[screen] wrote {out_json}\n")
    print(cmp_df.to_string(index=False))
    print(f"\n[screen] BEST -> {top.family} / {top.framing}  dev NDCG@{eval_k}={top.dev_ndcg:.4f}")
    print("[screen] 'significant' = paired-bootstrap 95% CI of the difference excludes 0.")


if __name__ == "__main__":
    main()
