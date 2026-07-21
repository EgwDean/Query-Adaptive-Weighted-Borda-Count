"""sections.py -- pipeline sections 4-9 (dataset build -> router -> benchmark).

Split out of pipeline.py purely for file size; `python src/pipeline.py` remains
the single entry point. See pipeline.py's header for the method and the section
list. Every function here takes (cfg, paths), writes its outputs, and returns a
one-line status; each skips if its outputs already exist.
"""

import os
import gc
import json
import time
import math
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import kendalltau

import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              ExtraTreesRegressor, ExtraTreesClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
import joblib

from utils import dataset_dir, processed_dir
from core import (N_THREADS, RRF_K, FUSERS, fuse_score, alpha_curve, ndcg, mrr,
                  recall_at, bootstrap_ci, paired_bootstrap, read_queries,
                  read_qrels, load_retrieval, fusion_arrays, topk_ids)

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

NON_FEATURE = {"dataset", "split", "qid", "alpha", "oracle_ndcg", "bm25_ndcg",
               "dense_ndcg", "alpha_sensitivity", "plateau_frac", "n_rel",
               "eval_k", "top_k"}
SCALE_SENSITIVE = {"elasticnet", "logreg", "mlp"}
NO_SAMPLE_WEIGHT = {"mlp"}
COST_CLASS = {"lookup": 1, "scores": 2, "embed": 3}


def resolve_splits(paths, name):
    """Which splits this dataset actually has, and which to fit / select on.

    Several BEIR datasets ship no `train` split (e.g. quora, dbpedia-entity are
    dev+test only). Falling back to dev as the FIT split is safe in frozen-config
    mode, where nothing is selected -- the router spec is inherited, so dev is
    only used to fit weights and the calibration table.
    """
    from utils import dataset_dir as _dd
    folder = _dd(paths, name)
    avail = [x for x in ("train", "dev", "test") if read_qrels(folder, x) is not None]
    fit = "train" if "train" in avail else ("dev" if "dev" in avail else None)
    sel = "dev" if ("dev" in avail and fit != "dev") else fit
    return avail, fit, sel


def run_tag(cfg):
    """Namespace for every section 4-9 output.

    Outputs from section 4 onward depend on the FUSION FUNCTION (the alpha label,
    the alpha->NDCG curve, and everything trained on them). Tagging keeps the
    minmax / rrf / borda arms of the study separate, and stops artefacts written
    by an earlier pipeline with a different schema from being silently reused.
    """
    fu = cfg["fusion"]
    return f"score-{fu['normalizer']}" if fu["function"] == "score" else fu["function"]


def feature_cost(n):
    if n.startswith(("autocorr", "apair_ratio", "query_centroid")):
        return "embed"
    if n.startswith(("top_score", "sigma_k", "margin", "norm_margin", "wig", "nqc",
                     "smv", "entropy", "robust_sigma", "jaccard", "kendall_tau",
                     "d_z", "d_wig")):
        return "scores"
    return "lookup"


# =========================================================================== #
# SECTION 4: features + alpha->NDCG curve + oracle label
# =========================================================================== #
def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _side_features(pfx, s, rows, emb, W, eps):
    """Score-distribution + embedding-coherence features for ONE retriever."""
    f = {}
    s = np.asarray(s, dtype=np.float64)
    sw = s[:W]
    mu, sd = sw.mean(), sw.std()
    f[f"top_score_{pfx}"] = float(s[0])
    f[f"sigma_k_{pfx}"] = float(sd)
    f[f"margin_{pfx}"] = float(s[0] - s[1]) if len(s) > 1 else 0.0
    f[f"norm_margin_{pfx}"] = float((s[0] - s[1]) / (abs(s[0]) + eps)) if len(s) > 1 else 0.0
    sp = np.maximum(sw, eps)
    mp = sp.mean()
    f[f"smv_{pfx}"] = float(((sp / mp) * np.abs(np.log(sp / mp))).mean())
    f[f"entropy_{pfx}"] = _entropy(_softmax(sw))
    t = max(1, int(0.1 * len(sw)))
    f[f"robust_sigma_{pfx}"] = float(np.sort(sw)[t:len(sw) - t].std()) if len(sw) - 2 * t > 1 else float(sd)
    f[f"zscore_top_{pfx}"] = float((s[0] - mu) / (sd + eps))
    f[f"zscore_margin_{pfx}"] = float((s[0] - s[1]) / (sd + eps)) if len(s) > 1 else 0.0
    # embedding coherence over a top-W window (one small matmul)
    w = min(W, len(rows))
    E = np.asarray(emb[np.asarray(rows[:w])], dtype=np.float32)
    sim = E @ E.T
    np.fill_diagonal(sim, 0.0)
    z = sw[:w] - sw[:w].mean()
    S0, den = sim.sum(), float((z * z).sum())
    f[f"autocorr_{pfx}"] = float((w / (S0 + eps)) * (z @ sim @ z) / (den + eps)) if S0 > 0 and den > 0 else 0.0
    top_mean = float(sim.sum() / (w * (w - 1))) if w > 1 else 0.0
    if len(rows) >= 2 * w:
        Eb = np.asarray(emb[np.asarray(rows[-w:])], dtype=np.float32)
        sb = Eb @ Eb.T
        np.fill_diagonal(sb, 0.0)
        bot = float(sb.sum() / (w * (w - 1))) if w > 1 else 0.0
    else:
        bot = top_mean
    f[f"apair_ratio_{pfx}"] = float(top_mean / (bot + eps))
    return f


def sec_dataset(cfg, paths):
    name = cfg["dataset"]
    tag = run_tag(cfg)
    fd = paths["feature_dataset"]
    splits = cfg["pipeline"]["splits"]
    top_k = cfg["retrieval"]["top_k"]
    eval_k = cfg["retrieval"].get("eval_k", 10)
    fu = cfg["fusion"]
    fusion, norm = fu["function"], fu["normalizer"]
    W = int(cfg["features"]["window"])
    eps = 1e-9
    alphas = np.round(np.arange(fu["alpha_min"], fu["alpha_max"] + 1e-9, fu["alpha_step"]), 4)
    N = top_k

    avail, _, _ = resolve_splits(paths, name)
    splits = [s for s in splits if s in avail]
    need = [s for s in splits
            if not (os.path.exists(os.path.join(fd, f"{name}_{tag}_{s}_features.csv")) and
                    os.path.exists(os.path.join(fd, f"{name}_{tag}_{s}_curve.npy")))]
    if not need:
        return f"already built for {splits}"

    folder = dataset_dir(paths, name)
    pdir = processed_dir(paths, name, create=False)
    with open(os.path.join(pdir, "corpus_ids.json"), encoding="utf-8") as f:
        cid = json.load(f)
    emb = np.load(os.path.join(pdir, "corpus_emb.npy"), mmap_mode="r")
    centroid = np.asarray(emb[::max(1, len(cid) // 200000)], dtype=np.float64).mean(0)
    centroid = centroid.astype(np.float32)
    np.save(os.path.join(fd, f"{name}_{tag}_alpha_grid.npy"), alphas.astype(np.float32))
    queries = read_queries(folder)

    for s in need:
        qr = read_qrels(folder, s)
        qids, bi, bv, di, dv, qe = load_retrieval(paths, name, s, top_k)
        rows, curves = [], []
        for i, q in enumerate(tqdm(qids, desc=f"  [{s}] features")):
            rels = {d: int(g) for d, g in qr.get(q, {}).items() if int(g) > 0}
            if not rels:
                continue
            bm_r, bm_s, dn_r, dn_s = bi[i], bv[i].astype(np.float64), di[i], dv[i].astype(np.float64)
            curve, a_star, nd = alpha_curve(bm_r, dn_r, bm_s, dn_s, rels, cid,
                                            alphas, N, eval_k, fusion, norm)
            if curve is None:
                continue
            r = {"dataset": name, "split": s, "qid": q}
            r["ql"] = float(len(queries[q].split()))
            qv = qe[i].astype(np.float64)
            r["query_centroid_cos"] = float(qv @ centroid /
                                            (np.linalg.norm(qv) * np.linalg.norm(centroid) + eps))
            r.update(_side_features("bm25", bm_s, bm_r, emb, W, eps))
            r.update(_side_features("dense", dn_s, dn_r, emb, W, eps))
            sb, sd_ = set(int(x) for x in bm_r), set(int(x) for x in dn_r)
            inter = sb & sd_
            r["jaccard"] = len(inter) / max(len(sb | sd_), 1)
            if len(inter) >= 2:
                rb = {int(d): j for j, d in enumerate(bm_r)}
                rd = {int(d): j for j, d in enumerate(dn_r)}
                c = list(inter)
                tau, _ = kendalltau([rb[d] for d in c], [rd[d] for d in c])
                r["kendall_tau"] = float(tau) if tau == tau else 0.0
            else:
                r["kendall_tau"] = 0.0
            for k_ in ("zscore_top", "zscore_margin", "entropy", "smv", "sigma_k"):
                r[f"d_{k_}"] = r[f"{k_}_bm25"] - r[f"{k_}_dense"]
            r.update({"alpha": a_star, "oracle_ndcg": nd,
                      "bm25_ndcg": ndcg([cid[int(j)] for j in bm_r], rels, eval_k) or 0.0,
                      "dense_ndcg": ndcg([cid[int(j)] for j in dn_r], rels, eval_k) or 0.0,
                      "alpha_sensitivity": float(curve.max() - curve.min()),
                      "plateau_frac": float(np.mean(curve >= curve.max() - 1e-6)),
                      "n_rel": len(rels), "eval_k": eval_k, "top_k": top_k})
            rows.append(r)
            curves.append(curve)
        pd.DataFrame(rows).to_csv(os.path.join(fd, f"{name}_{tag}_{s}_features.csv"), index=False)
        np.save(os.path.join(fd, f"{name}_{tag}_{s}_curve.npy"), np.stack(curves).astype(np.float32))
        print(f"  [{s}] {len(rows):,} rows x {len(rows[0])} cols")
    return f"built {need}"


# =========================================================================== #
# Router plumbing (shared by sections 5-9)
# =========================================================================== #
def load_split(paths, name, split, tag):
    fd = paths["feature_dataset"]
    return (pd.read_csv(os.path.join(fd, f"{name}_{tag}_{split}_features.csv")),
            np.load(os.path.join(fd, f"{name}_{tag}_{split}_curve.npy")))


def ndcg_of_alpha(pred, curve, grid):
    p = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    idx = np.abs(grid[None, :] - p[:, None]).argmin(axis=1)
    return curve[np.arange(curve.shape[0]), idx]


def make_target(a, framing, bins):
    if framing == "regression":
        return a.astype(np.float64)
    if framing == "binary":
        return (a > 0.5).astype(int)
    return np.abs(a[:, None] - bins[None, :]).argmin(axis=1)


def predict_alpha(est, X, framing, bins):
    if framing == "regression":
        return np.clip(est.predict(X), 0.0, 1.0)
    pr = est.predict_proba(X)
    cls = np.asarray(est.classes_)
    if framing == "binary":
        return pr[:, list(cls).index(1)] if 1 in cls else np.zeros(X.shape[0])
    return pr @ bins[cls]


def _scaler(t):
    k = t.suggest_categorical("scaler", ["standard", "robust", "quantile"])
    return {"standard": StandardScaler(), "robust": RobustScaler()}.get(
        k, QuantileTransformer(output_distribution="normal", subsample=100000, random_state=0))


def family_available(fam, fr):
    reg = fr == "regression"
    if fam == "lightgbm" and LGBMRegressor is None:
        return False
    if fam == "xgboost" and XGBRegressor is None:
        return False
    if fam == "catboost" and CatBoostRegressor is None:
        return False
    if fam == "elasticnet" and not reg:
        return False
    if fam == "logreg" and reg:
        return False
    return True


def build_estimator(t, fam, fr, seed, n_jobs=N_THREADS, mr=600):
    reg = fr == "regression"
    if fam == "lightgbm":
        p = dict(n_estimators=t.suggest_int("n_estimators", 100, mr, log=True),
                 learning_rate=t.suggest_float("learning_rate", .01, .3, log=True),
                 num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
                 min_child_samples=t.suggest_int("min_child_samples", 5, 100),
                 subsample=t.suggest_float("subsample", .6, 1.), subsample_freq=1,
                 colsample_bytree=t.suggest_float("colsample_bytree", .5, 1.),
                 reg_alpha=t.suggest_float("reg_alpha", 1e-8, 10., log=True),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-8, 10., log=True),
                 n_jobs=n_jobs, random_state=seed, verbose=-1)
        return LGBMRegressor(**p) if reg else LGBMClassifier(**p)
    if fam == "xgboost":
        p = dict(n_estimators=t.suggest_int("n_estimators", 100, mr, log=True),
                 learning_rate=t.suggest_float("learning_rate", .01, .3, log=True),
                 max_depth=t.suggest_int("max_depth", 3, 12),
                 min_child_weight=t.suggest_float("min_child_weight", 1., 20., log=True),
                 subsample=t.suggest_float("subsample", .6, 1.),
                 colsample_bytree=t.suggest_float("colsample_bytree", .5, 1.),
                 gamma=t.suggest_float("gamma", 1e-8, 5., log=True),
                 reg_alpha=t.suggest_float("reg_alpha", 1e-8, 10., log=True),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-8, 10., log=True),
                 n_jobs=n_jobs, random_state=seed, tree_method="hist", verbosity=0)
        return XGBRegressor(**p) if reg else XGBClassifier(**p)
    if fam == "catboost":
        p = dict(iterations=t.suggest_int("iterations", 100, mr, log=True),
                 learning_rate=t.suggest_float("learning_rate", .01, .3, log=True),
                 depth=t.suggest_int("depth", 4, 10),
                 l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1., 30., log=True),
                 thread_count=n_jobs, random_seed=seed, verbose=0, allow_writing_files=False)
        return CatBoostRegressor(**p) if reg else CatBoostClassifier(**p)
    if fam == "hist_gbdt":
        p = dict(max_iter=t.suggest_int("max_iter", 100, mr, log=True),
                 learning_rate=t.suggest_float("learning_rate", .01, .3, log=True),
                 max_leaf_nodes=t.suggest_int("max_leaf_nodes", 15, 255, log=True),
                 min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 100),
                 l2_regularization=t.suggest_float("l2_regularization", 1e-8, 10., log=True),
                 early_stopping=False, random_state=seed)
        return HistGradientBoostingRegressor(**p) if reg else HistGradientBoostingClassifier(**p)
    if fam in ("random_forest", "extra_trees"):
        p = dict(n_estimators=t.suggest_int("n_estimators", 100, mr, log=True),
                 max_depth=t.suggest_int("max_depth", 5, 30),
                 min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 20),
                 max_features=t.suggest_float("max_features", .2, 1.),
                 n_jobs=n_jobs, random_state=seed)
        if fam == "random_forest":
            return RandomForestRegressor(**p) if reg else RandomForestClassifier(**p)
        return ExtraTreesRegressor(**p) if reg else ExtraTreesClassifier(**p)
    if fam == "elasticnet":
        return ElasticNet(alpha=t.suggest_float("alpha", 1e-5, 10., log=True),
                          l1_ratio=t.suggest_float("l1_ratio", 0., 1.),
                          max_iter=5000, random_state=seed) if reg else None
    if fam == "logreg":
        if reg:
            return None
        pen = t.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
        p = dict(C=t.suggest_float("C", 1e-4, 100., log=True), penalty=pen,
                 solver="saga", max_iter=3000, n_jobs=n_jobs, random_state=seed)
        if pen == "elasticnet":
            p["l1_ratio"] = t.suggest_float("l1_ratio", 0., 1.)
        return LogisticRegression(**p)
    if fam == "mlp":
        w = t.suggest_categorical("width", [32, 64, 128, 256])
        p = dict(hidden_layer_sizes=tuple([w] * t.suggest_int("depth", 1, 3)),
                 alpha=t.suggest_float("alpha", 1e-6, 1e-1, log=True),
                 learning_rate_init=t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                 max_iter=t.suggest_int("max_iter", 200, 600), early_stopping=True,
                 n_iter_no_change=15, random_state=seed)
        return MLPRegressor(**p) if reg else MLPClassifier(**p)
    return None


def _fit(est, fam, use_w, X, y, w):
    if fam in SCALE_SENSITIVE:
        pipe = Pipeline([("scaler", est[0]), ("model", est[1])])
        pipe.fit(X, y, model__sample_weight=w) if use_w else pipe.fit(X, y)
        return pipe
    est.fit(X, y, sample_weight=w) if use_w else est.fit(X, y)
    return est


def fit_calibration(scores, curve, grid, n_bins):
    """HISTOGRAM BINNING: the model output is only a SCORE used to bin queries;
    each bin emits the alpha maximising ITS average NDCG curve. If the model has
    no signal every bin's curve equals the global one, so every bin picks the
    same alpha -> exactly the constant baseline. That is the floor."""
    edges = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    inner = np.unique(edges[1:-1])
    idx = np.digitize(scores, inner)
    g = grid[int(curve.mean(axis=0).argmax())]
    ba = np.full(len(inner) + 1, g)
    for b in range(len(inner) + 1):
        m = idx == b
        if m.sum() > 0:
            ba[b] = grid[int(curve[m].mean(axis=0).argmax())]
    return inner, ba


def apply_calibration(s, inner, ba):
    return ba[np.digitize(s, inner)]


def run_config(est, fam, fr, use_w, decision, n_cb, Xtr, ytr, wtr, tr_curve,
               Xev, bins, grid, seed, cf):
    if decision == "raw":
        m = _fit(est, fam, use_w, Xtr, ytr, wtr)
        return m, predict_alpha(m, Xev, fr, bins), None
    n = len(Xtr)
    perm = np.random.default_rng(seed).permutation(n)
    n_cal = min(max(int(cf * n), n_cb * 10), n - n_cb * 10)
    ci, fi = perm[:n_cal], perm[n_cal:]
    if fr != "regression" and len(np.unique(ytr[fi])) < 2:
        raise ValueError("degenerate split")
    m = _fit(est, fam, use_w, Xtr[fi], ytr[fi], wtr[fi])
    edges, ba = fit_calibration(predict_alpha(m, Xtr[ci], fr, bins),
                                tr_curve[ci], grid, n_cb)
    return m, apply_calibration(predict_alpha(m, Xev, fr, bins), edges, ba), (edges, ba)


def pred_diag(pred, a_true):
    p = np.clip(np.asarray(pred, float), 0, 1)
    c = 0.0 if p.std() < 1e-12 or np.std(a_true) < 1e-12 else float(np.corrcoef(p, a_true)[0, 1])
    return float(p.mean()), float(p.std()), c


def _router_ctx(cfg, paths, feats=None):
    """Shared setup: subsampled train matrices + dev matrices + grid."""
    name = cfg["dataset"]
    r = cfg["router"]
    seed = int(cfg.get("seed", 42))
    tag = run_tag(cfg)
    grid = np.load(os.path.join(paths["feature_dataset"],
                                f"{name}_{tag}_alpha_grid.npy")).astype(np.float64)
    _, fit_s, sel_s = resolve_splits(paths, name)
    tr, trc = load_split(paths, name, fit_s, tag)
    dv, dvc = load_split(paths, name, sel_s, tag)
    allf = [c for c in tr.columns if c not in NON_FEATURE]
    feats = feats or allf
    n = min(int(r["train_subset"]), len(tr))
    sub = np.random.default_rng(seed).choice(len(tr), size=n, replace=False)
    bins = np.linspace(0, 1, int(r["n_bins"]))
    return dict(name=name, tag=tag, seed=seed, fit_split=fit_s, sel_split=sel_s, grid=grid, bins=bins, allf=allf, feats=feats,
                Xtr=tr.iloc[sub][feats].to_numpy(float),
                alpha_tr=tr.iloc[sub]["alpha"].to_numpy(float),
                wtr=tr.iloc[sub]["alpha_sensitivity"].to_numpy(float),
                tr_curve=trc[sub], Xdev=dv[feats].to_numpy(float),
                alpha_dev=dv["alpha"].to_numpy(float), dv_curve=dvc,
                n_sub=n, n_boot=int(r["bootstrap_resamples"]),
                cf=float(r["calib_fraction"]), n_jobs=int(r.get("n_jobs", N_THREADS)),
                mr=int(r.get("max_boost_rounds", 600)),
                rules=list(r["decision_rules"]), cbins=list(r["n_calib_bins"]),
                trials=int(r["n_trials"]), tr_df=tr, dv_df=dv)


def _study(c, fam, fr, Xtr, Xdev, trials):
    ytr = make_target(c["alpha_tr"], fr, c["bins"])
    if fr != "regression" and len(np.unique(ytr)) < 2:
        return None

    def obj(t):
        est = build_estimator(t, fam, fr, c["seed"], c["n_jobs"], c["mr"])
        if est is None:
            raise optuna.TrialPruned()
        uw = fam not in NO_SAMPLE_WEIGHT and t.suggest_categorical("use_sample_weight", [True, False])
        if fam in SCALE_SENSITIVE:
            est = (_scaler(t), est)
        dec = t.suggest_categorical("decision_rule", c["rules"])
        ncb = t.suggest_categorical("n_calib_bins", c["cbins"]) if dec == "calibrated" else 0
        try:
            _, pr, _ = run_config(est, fam, fr, uw, dec, ncb, Xtr, ytr, c["wtr"],
                                  c["tr_curve"], Xdev, c["bins"], c["grid"], c["seed"], c["cf"])
        except Exception:
            raise optuna.TrialPruned()
        return float(ndcg_of_alpha(pr, c["dv_curve"], c["grid"]).mean())

    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=c["seed"]))
    st.optimize(obj, n_trials=trials, show_progress_bar=False)
    return st if [t for t in st.trials if t.value is not None] else None


def _refit_best(c, fam, fr, params, Xtr, Xdev):
    ytr = make_target(c["alpha_tr"], fr, c["bins"])
    est = build_estimator(optuna.trial.FixedTrial(params), fam, fr, c["seed"],
                          c["n_jobs"], c["mr"])
    if fam in SCALE_SENSITIVE:
        est = (_scaler(optuna.trial.FixedTrial(params)), est)
    return run_config(est, fam, fr, bool(params.get("use_sample_weight", False)),
                      params.get("decision_rule", "calibrated"),
                      int(params.get("n_calib_bins", 0)), Xtr, ytr, c["wtr"],
                      c["tr_curve"], Xdev, c["bins"], c["grid"], c["seed"], c["cf"])


def _references(c):
    a = int(np.argmax(c["tr_curve"].mean(axis=0)))
    return (c["dv_curve"][:, a], c["dv_curve"].max(axis=1), float(c["grid"][a]))


# =========================================================================== #
# SECTION 5: screen families x framings
# =========================================================================== #
def sec_screen(cfg, paths):
    tag = run_tag(cfg)
    out = os.path.join(paths["router_screening"], f"{cfg['dataset']}_{tag}_screen.csv")
    if os.path.exists(out):
        d = pd.read_csv(out)
        if "family" in d.columns:
            b = d[d.family != "reference"].iloc[0]
            return f"already done: best {b['family']}|{b['framing']} {b['dev_ndcg']:.4f}"
        print(f"  {out}: unexpected schema -- rebuilding.")
    c = _router_ctx(cfg, paths)
    r = cfg["router"]
    const_pq, oracle_pq, a_const = _references(c)
    print(f"  {len(c['feats'])} features | train_sub={c['n_sub']:,} dev={len(c['Xdev']):,}")
    print(f"  constant alpha={a_const:.2f} -> {const_pq.mean():.4f} | oracle {oracle_pq.mean():.4f}")

    recs = [dict(family="reference", framing="constant_alpha", params=json.dumps({"alpha": a_const}),
                 dev_ndcg=float(const_pq.mean()), pred_std=0.0, corr=0.0),
            dict(family="reference", framing="oracle", params="{}",
                 dev_ndcg=float(oracle_pq.mean()), pred_std=0.0, corr=1.0)]
    for fam in r["families"]:
        for fr in r["framings"]:
            if not family_available(fam, fr):
                continue
            t = time.perf_counter()
            st = _study(c, fam, fr, c["Xtr"], c["Xdev"], c["trials"])
            if st is None:
                continue
            _, pred, _ = _refit_best(c, fam, fr, st.best_trial.params, c["Xtr"], c["Xdev"])
            pq = ndcg_of_alpha(pred, c["dv_curve"], c["grid"])
            pm, ps, pc = pred_diag(pred, c["alpha_dev"])
            recs.append(dict(family=fam, framing=fr, params=json.dumps(st.best_trial.params),
                             dev_ndcg=float(st.best_value), pred_std=ps, corr=pc))
            print(f"  {fam}|{fr:11s} {st.best_value:.4f} std={ps:.3f} "
                  f"({time.perf_counter()-t:.0f}s)")
    df = pd.DataFrame(recs).sort_values("dev_ndcg", ascending=False).reset_index(drop=True)
    df.to_csv(out, index=False)
    b = df[df.family != "reference"].iloc[0]
    with open(os.path.join(paths["router_screening"], f"{cfg['dataset']}_{tag}_screen_best.json"),
              "w", encoding="utf-8") as f:
        json.dump(dict(family=b["family"], framing=b["framing"],
                       params=json.loads(b["params"]), dev_ndcg=float(b["dev_ndcg"]),
                       features=c["feats"]), f, indent=2)
    return f"best {b['family']}|{b['framing']} {b['dev_ndcg']:.4f}"


# =========================================================================== #
# SECTION 6: greedy backward feature ablation
# =========================================================================== #
def sec_ablate(cfg, paths):
    name = cfg["dataset"]
    tag = run_tag(cfg)
    out = os.path.join(paths["router_screening"], f"{name}_{tag}_ablation.csv")
    if os.path.exists(out):
        d = pd.read_csv(out)
        if "chosen" in d.columns and d["chosen"].any():
            return f"already done: chose {int(d[d.chosen].iloc[0]['n_features'])} features"
        print(f"  {out}: unexpected schema -- rebuilding.")
    ab = cfg["ablation"]
    c = _router_ctx(cfg, paths)
    fam, fr, params = ab["family"], ab["framing"], ab["params"]
    minf = int(ab["min_features"])
    fidx = {f: i for i, f in enumerate(c["allf"])}
    n0 = len(c["allf"])
    total = sum(range(minf + 1, n0 + 1)) + 1
    print(f"  workhorse {fam}|{fr} | pruning {n0} -> {minf} = {total:,} fits")

    def ev(fs):
        i = [fidx[f] for f in fs]
        _, pr, _ = _refit_best(c, fam, fr, params, c["Xtr"][:, i], c["Xdev"][:, i])
        return ndcg_of_alpha(pr, c["dv_curve"], c["grid"])

    t0 = time.perf_counter()
    base = ev(c["allf"])
    fit = 1
    rows = [dict(n_features=n0, dropped="-", cost="-", dev_ndcg=float(base.mean()),
                 features="|".join(c["allf"]))]
    cur, pqs = list(c["allf"]), {n0: base}
    while len(cur) > minf:
        cands = []
        for f in cur:
            pq = ev([x for x in cur if x != f])
            fit += 1
            cands.append((float(pq.mean()), f, pq))
            el = time.perf_counter() - t0
            print(f"\r  [{fit}/{total}] n={len(cur)} drop {f[:26]:26s} {pq.mean():.4f} "
                  f"| {el/60:.1f}m ETA {el/fit*(total-fit)/60:.1f}m", end="", flush=True)
        best = max(c0[0] for c0 in cands)
        tied = [x for x in cands if best - x[0] < 1e-9]
        sc, sf, spq = max(tied, key=lambda x: COST_CLASS[feature_cost(x[1])])
        cur = [x for x in cur if x != sf]
        pqs[len(cur)] = spq
        rows.append(dict(n_features=len(cur), dropped=sf, cost=feature_cost(sf),
                         dev_ndcg=sc, features="|".join(cur)))
    print()
    path = pd.DataFrame(rows)
    bi = int(path.loc[path.dev_ndcg.idxmax(), "n_features"])
    bpq = pqs[bi]
    tied = []
    for _, r0 in path.iterrows():
        _, lo, hi = paired_bootstrap(pqs[int(r0["n_features"])], bpq, c["n_boot"], c["seed"])
        tied.append(not (lo > 0 or hi < 0))
    path["tied_with_best"] = tied
    path["chosen"] = False
    ci = path[path.tied_with_best].sort_values("n_features").index[0]
    path.loc[ci, "chosen"] = True
    path.to_csv(out, index=False)
    ch = path.loc[ci]
    with open(os.path.join(paths["router_screening"], f"{name}_{tag}_ablation_best.json"),
              "w", encoding="utf-8") as f:
        json.dump(dict(n_features=int(ch["n_features"]), dev_ndcg=float(ch["dev_ndcg"]),
                       features=ch["features"].split("|"),
                       full_ndcg=float(path.iloc[0]["dev_ndcg"])), f, indent=2)
    return (f"{n0} -> {int(ch['n_features'])} features, {path.iloc[0]['dev_ndcg']:.4f} "
            f"-> {ch['dev_ndcg']:.4f}")


# =========================================================================== #
# SECTION 7: re-screen families x feature sets
# =========================================================================== #
def sec_rescreen(cfg, paths):
    name = cfg["dataset"]
    tag = run_tag(cfg)
    out = os.path.join(paths["router_screening"], f"{name}_{tag}_rescreen.csv")
    if os.path.exists(out):
        d = pd.read_csv(out)
        if "chosen" in d.columns and d["chosen"].any():
            b = d[d.chosen].iloc[0]
            return f"already done: chose {b['config']} {b['dev_ndcg']:.4f}"
        print(f"  {out}: unexpected schema -- rebuilding.")
    rc = cfg["rescreen"]
    c = _router_ctx(cfg, paths)
    ap = pd.read_csv(os.path.join(paths["router_screening"], f"{name}_{tag}_ablation.csv"))
    by_n = {int(r["n_features"]): r["features"].split("|") for _, r in ap.iterrows()}
    sets = {}
    for s0 in rc["feature_set_sizes"]:
        s0 = int(s0)
        if s0 <= 0 or s0 >= len(c["allf"]):        # 0 (or too big) -> the full set
            sets[len(c["allf"])] = list(c["allf"])
        elif by_n.get(s0):
            sets[s0] = by_n[s0]
    sets = {k: v for k, v in sets.items() if v}
    fidx = {f: i for i, f in enumerate(c["allf"])}
    const_pq, oracle_pq, a_const = _references(c)
    per_q, meta = {"constant": const_pq, "oracle": oracle_pq}, {}
    meta["constant"] = dict(n=0, cost=0, std=0.0)
    meta["oracle"] = dict(n=0, cost=0, std=0.0)
    n_st = sum(1 for f in sets for fam in rc["families"] for fr in rc["framings"]
               if family_available(fam, fr))
    print(f"  sets {sorted(sets)} | {n_st} studies x {rc['n_trials']} trials")
    done = 0
    t0 = time.perf_counter()
    for nf in sorted(sets):
        fs = sets[nf]
        i = [fidx[f] for f in fs]
        Xtr, Xdev = c["Xtr"][:, i], c["Xdev"][:, i]
        cost = max(COST_CLASS[feature_cost(f)] for f in fs)
        for fam in rc["families"]:
            for fr in rc["framings"]:
                if not family_available(fam, fr):
                    continue
                st = _study(c, fam, fr, Xtr, Xdev, int(rc["n_trials"]))
                done += 1
                if st is None:
                    continue
                _, pred, _ = _refit_best(c, fam, fr, st.best_trial.params, Xtr, Xdev)
                pq = ndcg_of_alpha(pred, c["dv_curve"], c["grid"])
                pm, ps, pc = pred_diag(pred, c["alpha_dev"])
                key = f"f{nf}|{fam}|{fr}"      # NOT `tag`: that is the fusion tag
                per_q[key] = pq
                meta[key] = dict(n=nf, cost=cost, std=ps, family=fam, framing=fr,
                                 params=st.best_trial.params, features=fs)
                el = time.perf_counter() - t0
                print(f"  [{done}/{n_st}] {key:32s} {pq.mean():.4f} std={ps:.3f} "
                      f"ETA {el/done*(n_st-done)/60:.1f}m")
    models = {k: v for k, v in per_q.items() if k not in ("constant", "oracle")}
    top = max(models, key=lambda k: models[k].mean())
    rows = []
    for key, pq in per_q.items():      # NOT `tag`: that is the fusion tag
        d_, lo, hi = paired_bootstrap(pq, per_q[top], c["n_boot"], c["seed"])
        m = meta[key]
        rows.append(dict(config=key, n_features=m["n"], max_cost=m["cost"],
                         dev_ndcg=float(pq.mean()), diff_vs_best=d_,
                         tied_with_best=not (lo > 0 or hi < 0), pred_std=m["std"]))
    df = pd.DataFrame(rows).sort_values("dev_ndcg", ascending=False).reset_index(drop=True)
    df["chosen"] = False
    cand = df[df.tied_with_best & (df.n_features > 0)]
    ci = cand.sort_values(["n_features", "max_cost", "dev_ndcg"],
                          ascending=[True, True, False]).index[0]
    df.loc[ci, "chosen"] = True
    df.to_csv(out, index=False)
    ch = meta[df.loc[ci, "config"]]
    with open(os.path.join(paths["router_screening"], f"{name}_{tag}_rescreen_best.json"),
              "w", encoding="utf-8") as f:
        json.dump(dict(config=df.loc[ci, "config"], family=ch["family"],
                       framing=ch["framing"], params=ch["params"],
                       features=ch["features"], n_features=ch["n"],
                       dev_ndcg=float(df.loc[ci, "dev_ndcg"]), nominal_best=top,
                       nominal_best_ndcg=float(models[top].mean())), f, indent=2)
    return f"chose {df.loc[ci,'config']} {df.loc[ci,'dev_ndcg']:.4f} (nominal best {top})"


# =========================================================================== #
# SECTION 8: final fit on the full train split -> FREEZE
# =========================================================================== #
def sec_final_fit(cfg, paths):
    name = cfg["dataset"]
    tag = run_tag(cfg)
    art = os.path.join(paths["router_final"], f"{name}_{tag}_router.joblib")
    if os.path.exists(art):
        return f"already frozen: {art}"
    # Held-out datasets inherit the spec chosen on the DEVELOPMENT dataset, so
    # no selection whatsoever happens on them (see study.inherit_spec_from).
    src_ds = (cfg.get("study", {}) or {}).get("inherit_spec_from") or name
    spec_p = os.path.join(paths["router_screening"], f"{src_ds}_{tag}_rescreen_best.json")
    if not os.path.exists(spec_p) and src_ds != name:
        raise SystemExit(f"[final] {spec_p} missing -- run the development dataset "
                         f"({src_ds}) through section 7 first.")
    with open(spec_p, encoding="utf-8") as f:
        spec = json.load(f)
    if src_ds != name:
        print(f"  inheriting router spec from '{src_ds}' ({spec['family']}|{spec['framing']}, "
              f"{len(spec['features'])} features)")
    c = _router_ctx(cfg, paths, feats=spec["features"])
    r = cfg["router"]
    tr, trc = load_split(paths, name, c["fit_split"], tag)
    dv, dvc = load_split(paths, name, c["sel_split"], tag)
    feats = spec["features"]
    Xtr = tr[feats].to_numpy(float)
    ytr = make_target(tr["alpha"].to_numpy(float), spec["framing"], c["bins"])
    wtr = tr["alpha_sensitivity"].to_numpy(float)
    Xdev = dv[feats].to_numpy(float)
    p = spec["params"]
    est = build_estimator(optuna.trial.FixedTrial(p), spec["family"], spec["framing"],
                          c["seed"], c["n_jobs"], c["mr"])
    if spec["family"] in SCALE_SENSITIVE:
        est = (_scaler(optuna.trial.FixedTrial(p)), est)
    model, pred, calib = run_config(est, spec["family"], spec["framing"],
                                    bool(p.get("use_sample_weight", False)),
                                    p.get("decision_rule", "calibrated"),
                                    int(p.get("n_calib_bins", 20) or 20),
                                    Xtr, ytr, wtr, trc, Xdev, c["bins"], c["grid"],
                                    c["seed"], c["cf"])
    pq = ndcg_of_alpha(pred, dvc, c["grid"])
    a = int(np.argmax(trc.mean(axis=0)))
    d_, lo, hi = paired_bootstrap(pq, dvc[:, a], c["n_boot"], c["seed"])
    joblib.dump(dict(model=model, family=spec["family"], framing=spec["framing"],
                     features=feats, bins=c["bins"],
                     decision=p.get("decision_rule", "calibrated"),
                     calib_edges=(calib[0] if calib else None),
                     calib_bin_alpha=(calib[1] if calib else None),
                     alpha_grid=c["grid"]), art)
    with open(os.path.join(paths["router_final"], f"{name}_{tag}_router_meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(dict(dataset=name, **{k: spec[k] for k in ("family", "framing", "params")},
                       features=feats, n_train=len(Xtr), dev_ndcg=float(pq.mean()),
                       dev_constant=float(dvc[:, a].mean()),
                       dev_oracle=float(dvc.max(1).mean()),
                       gain_vs_constant=d_, gain_ci=[lo, hi],
                       note="FROZEN; section 9 opens test once."), f, indent=2)
    return (f"dev {pq.mean():.4f} vs constant {dvc[:,a].mean():.4f} "
            f"({d_:+.4f} CI [{lo:+.4f},{hi:+.4f}]) -> frozen")


# =========================================================================== #
# SECTION 9: benchmark on TEST (once)
# =========================================================================== #
def sec_benchmark(cfg, paths):
    name = cfg["dataset"]
    tag = run_tag(cfg)
    out = os.path.join(paths["router_final"], f"{name}_{tag}_benchmark.csv")
    if os.path.exists(out):
        return f"already done: {out}"
    top_k = cfg["retrieval"]["top_k"]
    eval_k = cfg["retrieval"].get("eval_k", 10)
    fu = cfg["fusion"]
    norm = fu["normalizer"]
    N = top_k
    seed = int(cfg.get("seed", 42))
    n_boot = int(cfg["router"]["bootstrap_resamples"])
    alphas = np.round(np.arange(fu["alpha_min"], fu["alpha_max"] + 1e-9, fu["alpha_step"]), 4)
    folder = dataset_dir(paths, name)
    pdir = processed_dir(paths, name, create=False)
    with open(os.path.join(pdir, "corpus_ids.json"), encoding="utf-8") as f:
        cid = json.load(f)
    R = joblib.load(os.path.join(paths["router_final"], f"{name}_{tag}_router.joblib"))

    lists = {}
    for s in ("dev", "test"):
        qids, bi, bv, di, dv_, _ = load_retrieval(paths, name, s, top_k)
        qr = read_qrels(folder, s)
        lists[s] = (qids, qr, bi, bv, di, dv_)

    # tune every global alpha on DEV -- same opportunity the router had
    def tune(fusion, nrm):
        """Grid-search ONE global alpha on dev. Precomputes each query's fusion
        arrays once, so the alpha sweep is a vectorised weighted sum."""
        qids, qr, bi, bv, di, dv_ = lists["dev"]
        pre = []
        for i, q in enumerate(qids):
            rels = {d: int(g) for d, g in qr.get(q, {}).items() if int(g) > 0}
            if rels:
                pre.append((fusion_arrays(bi[i], di[i], bv[i].astype(float),
                                          dv_[i].astype(float), fusion, N, nrm), rels))
        tot = np.zeros(len(alphas))
        for (docs, va, vb), rels in pre:
            for j, a in enumerate(alphas):
                v = ndcg([cid[d] for d in topk_ids(docs, a * va + (1 - a) * vb, eval_k)],
                         rels, eval_k)
                tot[j] += v or 0.0
        return float(alphas[int(tot.argmax())])
    print("  tuning global alphas on dev ...")
    a_sc, a_bo, a_rr = tune("score", norm), tune("borda", None), tune("rrf", None)
    print(f"  score a*={a_sc:.2f} | borda a*={a_bo:.2f} | rrf a*={a_rr:.2f}")

    # dtype=str: numeric qids (fever/fiqa/quora) would otherwise be read as int64
    # and never match the string qids from retrieval, giving an empty feature set.
    tdf = pd.read_csv(os.path.join(paths["feature_dataset"],
                                   f"{name}_{tag}_test_features.csv"),
                      dtype={"qid": str}).set_index("qid")
    qids, qr, bi, bv, di, dv_ = lists["test"]
    keep = [q for q in qids if q in tdf.index]
    if not keep:
        raise SystemExit(f"[benchmark] no test qids matched {name}_{tag} features "
                         f"(retrieval vs feature-CSV qid mismatch).")
    X = tdf.loc[keep, R["features"]].to_numpy(float)
    t = time.perf_counter()
    raw = predict_alpha(R["model"], X, R["framing"], R["bins"])
    if R["decision"] == "calibrated" and R["calib_edges"] is not None:
        raw = apply_calibration(raw, R["calib_edges"], R["calib_bin_alpha"])
    us = (time.perf_counter() - t) / max(len(keep), 1) * 1e6
    ralpha = dict(zip(keep, raw))
    print(f"  router: {us:.1f} us/query, alpha mean={raw.mean():.3f} std={raw.std():.3f}")

    # The ROUTER, the primary baseline, and the oracle all use THIS cell's fusion
    # function (the one the router was trained on) -- applying a borda/rrf-trained
    # router through score fusion, or comparing it to a score baseline, is
    # meaningless. The other two fusions appear as cross-context static rows.
    prim = fu["function"]
    prim_norm = norm if prim == "score" else None
    a_star = {"score": a_sc, "borda": a_bo, "rrf": a_rr}
    label_of = {"score": "Score fusion", "borda": "Borda", "rrf": "RRF"}

    # (label, kind, (fusion, alpha, normalizer))
    METHODS = [("BM25", "bm25", None), ("Dense", "dense", None),
               (f"{label_of[prim]} a=0.5", "fuse", (prim, 0.5, prim_norm)),
               (f"{label_of[prim]} a*={a_star[prim]:.2f} [BASELINE]", "fuse",
                (prim, a_star[prim], prim_norm))]
    for other in ("score", "borda", "rrf"):     # cross-fusion static context
        if other != prim:
            METHODS.append((f"{label_of[other]} a*={a_star[other]:.2f}", "fuse",
                            (other, a_star[other], norm if other == "score" else None)))
    METHODS += [("ROUTER (ours)", "router", (prim, None, prim_norm)),
                ("Oracle alpha", "oracle", (prim, None, prim_norm))]

    acc = {m[0]: {k: [] for k in ("ndcg10", "ndcg100", "mrr100", "recall100")} for m in METHODS}
    used = []
    for i, q in enumerate(tqdm(qids, desc="  scoring test")):
        rels = {d: int(g) for d, g in qr.get(q, {}).items() if int(g) > 0}
        if not rels:
            continue
        used.append(q)
        b_r, b_s, d_r, d_s = bi[i], bv[i].astype(float), di[i], dv_[i].astype(float)
        for label, kind, spec in METHODS:
            if kind == "bm25":
                rk = [cid[int(j)] for j in b_r]
            elif kind == "dense":
                rk = [cid[int(j)] for j in d_r]
            elif kind == "oracle":
                ff, _, nn = spec
                docs, va, vb = fusion_arrays(b_r, d_r, b_s, d_s, ff, N, nn)
                best, ba = -1.0, alphas[0]
                for aa in alphas:
                    v = ndcg([cid[d] for d in topk_ids(docs, aa * va + (1 - aa) * vb, eval_k)],
                             rels, eval_k)
                    if v is not None and v > best:
                        best, ba = v, aa
                rk = [cid[j] for j in FUSERS[ff](b_r, d_r, b_s, d_s, ba, N, nn)]
            else:                                # "fuse" (static) or "router"
                ff, a, nn = spec
                aa = ralpha.get(q, a_star[prim]) if kind == "router" else a
                rk = [cid[j] for j in FUSERS[ff](b_r, d_r, b_s, d_s, aa, N, nn)]
            acc[label]["ndcg10"].append(ndcg(rk, rels, eval_k) or 0.0)
            acc[label]["ndcg100"].append(ndcg(rk, rels, 100) or 0.0)
            acc[label]["mrr100"].append(mrr(rk, rels, 100))
            acc[label]["recall100"].append(recall_at(rk, rels, 100) or 0.0)

    base_lbl = [m[0] for m in METHODS if "[BASELINE]" in m[0]][0]
    base = np.asarray(acc[base_lbl]["ndcg10"])
    rows = []
    for label, _, _ in METHODS:
        a10 = np.asarray(acc[label]["ndcg10"])
        lo, hi = bootstrap_ci(a10, n_boot, seed)
        d_, dlo, dhi = paired_bootstrap(a10, base, n_boot, seed)
        rows.append(dict(method=label, ndcg10=a10.mean(), ci_lo=lo, ci_hi=hi,
                         ndcg100=np.mean(acc[label]["ndcg100"]),
                         mrr100=np.mean(acc[label]["mrr100"]),
                         recall100=np.mean(acc[label]["recall100"]),
                         diff_vs_baseline=d_, diff_ci_lo=dlo, diff_ci_hi=dhi,
                         significant=bool(dlo > 0 or dhi < 0)))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.DataFrame({"qid": used, **{l: acc[l]["ndcg10"] for l, _, _ in METHODS}}).to_csv(
        os.path.join(paths["router_final"], f"{name}_{tag}_benchmark_per_query.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n" + df[["method", "ndcg10", "ci_lo", "ci_hi", "ndcg100", "mrr100",
                     "recall100", "diff_vs_baseline", "significant"]].to_string(index=False))
    r0 = df[df.method == "ROUTER (ours)"].iloc[0]
    with open(os.path.join(paths["router_final"], f"{name}_{tag}_benchmark.json"),
              "w", encoding="utf-8") as f:
        json.dump(dict(dataset=name, n_queries=len(used), baseline=base_lbl,
                       router_ndcg10=float(r0["ndcg10"]),
                       gain=float(r0["diff_vs_baseline"]),
                       gain_ci=[float(r0["diff_ci_lo"]), float(r0["diff_ci_hi"])],
                       significant=bool(r0["significant"]), router_us_per_query=us,
                       table=df.to_dict("records")), f, indent=2)
    return (f"ROUTER {r0['ndcg10']:.4f} vs baseline {r0['diff_vs_baseline']:+.4f} "
            f"{'SIGNIFICANT' if r0['significant'] else 'n.s.'} -- TEST IS NOW SPENT")
