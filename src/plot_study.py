"""Analysis figures for the study (H1 scaling, H2 decision rule, H3 invariance).

    python src/plot_study.py            # -> data/results/figures/*.png

Reads STUDY_SUMMARY.csv and the H2 table. make_slides.py imports these builders
so the deck and the standalone figures never diverge.

Colour convention, used in every scatter/bar here:
    green = significant gain      red = significant loss      grey = not significant
(significance = paired bootstrap of the per-query NDCG@10 difference vs the best
constant alpha, on test, 95% CI excluding zero.)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils import load_config, get_paths

GREEN, RED, GREY, BLUE, GOLD = "#2e7d32", "#c62828", "#9aa0a6", "#0f4c81", "#c9a227"
FUSIONS = ["score-minmax", "rrf", "borda"]
FLABEL = {"score-minmax": "score fusion", "rrf": "RRF", "borda": "Borda"}
DS_ORDER = ["hotpotqa", "fever", "nfcorpus", "scifact", "fiqa", "quora"]


def point_color(row):
    if not row.significant:
        return GREY
    return GREEN if row.gain > 0 else RED


def sig_legend(ax, loc="upper left", extra=None):
    h = [Line2D([], [], marker="o", ls="", color=GREEN, label="significant gain"),
         Line2D([], [], marker="o", ls="", color=RED, label="significant loss"),
         Line2D([], [], marker="o", ls="", color=GREY, label="not significant")]
    if extra:
        h += extra
    ax.legend(handles=h, loc=loc, fontsize=8, framealpha=0.95)


def _annotate(ax, x, y, text, fs=7.5):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(6, 4),
                fontsize=fs, color="#1a1a2e")


def _fit_line(ax, x, y, color=BLUE):
    if len(x) < 2 or np.ptp(x) == 0:
        return np.nan
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(min(x), max(x), 50)
    ax.plot(xs, a + b * xs, color=color, lw=1.6, alpha=0.85, zorder=1)
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------------------------------------------------------- #
# H1
# --------------------------------------------------------------------------- #
def fig_h1_combined(summ, out):
    """All held-out cells, every point labelled, dev shown separately."""
    held = summ[summ.role == "held-out"]
    dev = summ[summ.role == "dev"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for _, r in held.iterrows():
        ax.scatter(r.alpha_iqr, r.gain, c=point_color(r), s=80, zorder=3,
                   edgecolor="white", lw=0.7)
        _annotate(ax, r.alpha_iqr, r.gain, f"{r.dataset}·{FLABEL[r.fusion].split()[0]}")
    for _, r in dev.iterrows():
        ax.scatter(r.alpha_iqr, r.gain, c=point_color(r), s=95, marker="X", zorder=3,
                   edgecolor="black", lw=0.8)
        _annotate(ax, r.alpha_iqr, r.gain, f"{r.dataset}·{FLABEL[r.fusion].split()[0]} (dev)")
    rr = _fit_line(ax, held.alpha_iqr.to_numpy(), held.gain.to_numpy())
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("oracle-α spread (IQR, non-test split)  —  retriever complementarity")
    ax.set_ylabel("NDCG@10 gain vs best constant α")
    ax.set_title(f"H1: gain grows with complementarity   "
                 f"(held-out only: Pearson r = {rr:+.3f}, n = {len(held)})")
    sig_legend(ax, extra=[Line2D([], [], marker="X", ls="", color="black",
                                 label="development dataset (excluded from fit)")])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return rr


def fig_h1_by_fusion(summ, out):
    """One panel per fusion: does H1 hold in each fusion separately? (H3 support)"""
    held = summ[summ.role == "held-out"]
    dev = summ[summ.role == "dev"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    rs = {}
    for ax, fu in zip(axes, FUSIONS):
        h = held[held.fusion == fu]
        d = dev[dev.fusion == fu]
        for _, r in h.iterrows():
            ax.scatter(r.alpha_iqr, r.gain, c=point_color(r), s=85, zorder=3,
                       edgecolor="white", lw=0.7)
            _annotate(ax, r.alpha_iqr, r.gain, r.dataset, fs=8.5)
        for _, r in d.iterrows():
            ax.scatter(r.alpha_iqr, r.gain, c=point_color(r), s=100, marker="X",
                       zorder=3, edgecolor="black", lw=0.8)
            _annotate(ax, r.alpha_iqr, r.gain, f"{r.dataset} (dev)", fs=8.5)
        rr = _fit_line(ax, h.alpha_iqr.to_numpy(), h.gain.to_numpy())
        rs[fu] = rr
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{FLABEL[fu]}   (r = {rr:+.3f}, n = {len(h)})", fontsize=12)
        ax.set_xlabel("oracle-α spread (IQR)")
    axes[0].set_ylabel("NDCG@10 gain vs best constant α")
    sig_legend(axes[0])
    fig.suptitle("H1 holds within every fusion function", fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return rs


# --------------------------------------------------------------------------- #
# H3
# --------------------------------------------------------------------------- #
def fig_gain_by_fusion(summ, out):
    """Grouped bars: gain per dataset under each fusion, with CIs and sig stars."""
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    w = 0.26
    x = np.arange(len(DS_ORDER))
    hatch = {"score-minmax": "", "rrf": "//", "borda": ".."}
    for j, fu in enumerate(FUSIONS):
        sub = summ[summ.fusion == fu].set_index("dataset").reindex(DS_ORDER)
        off = (j - 1) * w
        cols = [point_color(r) if r.notna().all() else GREY for _, r in sub.iterrows()]
        err = np.vstack([sub.gain - sub.gain_ci_lo, sub.gain_ci_hi - sub.gain])
        ax.bar(x + off, sub.gain, w, color=cols, yerr=err, capsize=3,
               edgecolor="black", lw=0.5, hatch=hatch[fu])
        for xi, (_, r) in zip(x + off, sub.iterrows()):
            if r.significant:
                ax.annotate("★", (xi, r.gain), ha="center", fontsize=9,
                            textcoords="offset points",
                            xytext=(0, 11 if r.gain >= 0 else -17))
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\nIQR={summ[(summ.dataset==d)&(summ.fusion=='score-minmax')].alpha_iqr.iloc[0]:.2f}"
                        for d in DS_ORDER], fontsize=9)
    ax.set_ylabel("NDCG@10 gain vs best constant α")
    ax.set_title("H3: same pattern under all three fusion functions  (★ = significant)")
    style = [Line2D([], [], color="black", lw=6, alpha=0.25, label=FLABEL[f]) for f in FUSIONS]
    sig_legend(ax, loc="upper right")
    ax2 = ax.twinx(); ax2.axis("off")
    ax2.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                                      hatch=hatch[f], label=FLABEL[f]) for f in FUSIONS],
               loc="lower right", fontsize=8, framealpha=0.95, title="fusion")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_headroom_by_fusion(summ, out):
    """How much of the static->oracle headroom the router captures, per fusion."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, fu in zip(axes, FUSIONS):
        sub = summ[summ.fusion == fu].set_index("dataset").reindex(DS_ORDER)
        x = np.arange(len(DS_ORDER))
        ax.bar(x - 0.26, sub.static_best, 0.26, label="constant α", color=GREY)
        ax.bar(x, sub.router, 0.26, label="router", color=BLUE)
        ax.bar(x + 0.26, sub.oracle, 0.26, label="oracle", color=GOLD)
        ax.set_xticks(x); ax.set_xticklabels(DS_ORDER, rotation=25, ha="right", fontsize=9)
        ax.set_title(FLABEL[fu], fontsize=12)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("NDCG@10")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.suptitle("Constant vs router vs oracle ceiling, by fusion", fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# H2
# --------------------------------------------------------------------------- #
def fig_h2(h2, out):
    m = h2[h2.family != "reference"]
    const = float(h2[h2.framing == "constant"].dev_ndcg.iloc[0])
    raw = m[m.rule == "raw"].reset_index(drop=True)
    cal = m[m.rule == "calibrated"].reset_index(drop=True)
    j = raw.merge(cal, on=["family", "framing"], suffixes=("_raw", "_cal"))
    x = np.arange(len(j))
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.axhline(const, ls="--", color="black", lw=1.4)
    ax.vlines(x, j.dev_ndcg_raw, j.dev_ndcg_cal, color="#bbbbbb", lw=1.2, zorder=1)
    ax.scatter(x, j.dev_ndcg_raw, c=RED, s=60, zorder=3, label="raw output used as α")
    ax.scatter(x, j.dev_ndcg_cal, c=GREEN, marker="s", s=60, zorder=3, label="calibrated")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n{fr}" for f, fr in zip(j.family, j.framing)],
                       rotation=45, ha="right", fontsize=7.5)
    ax.annotate(f"best constant α = {const:.4f}", (len(j) - 0.5, const),
                textcoords="offset points", xytext=(-4, 6), ha="right", fontsize=9)
    ax.set_ylabel("dev NDCG@10")
    ax.set_title(f"H2: every raw-output router falls below the constant; "
                 f"every calibrated one clears it ({len(j)}/{len(j)})")
    ax.legend(loc="center right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# best alpha vs mean alpha
# --------------------------------------------------------------------------- #
def _sel_split(fd, ds, tag):
    """The split alpha* is tuned on: dev when present, else train."""
    for s in ("dev", "train"):
        if os.path.exists(os.path.join(fd, f"{ds}_{tag}_{s}_features.csv")):
            return s
    return None


def alpha_star_table(paths):
    """Per cell: the mean/median of the per-query oracle alphas, the alpha that
    actually maximises mean NDCG, and the NDCG lost by using the mean instead.

    argmax of the mean curve is not the mean of the per-query argmaxes, so
    averaging oracle alphas is not a valid way to pick a constant.
    """
    fd = paths["feature_dataset"]
    rows = []
    for ds in DS_ORDER:
        for fu in FUSIONS:
            s = _sel_split(fd, ds, fu)
            if s is None:
                continue
            fp = os.path.join(fd, f"{ds}_{fu}_{s}_features.csv")
            cp = os.path.join(fd, f"{ds}_{fu}_{s}_curve.npy")
            gp = os.path.join(fd, f"{ds}_{fu}_alpha_grid.npy")
            if not (os.path.exists(cp) and os.path.exists(gp)):
                continue
            a = pd.read_csv(fp, usecols=["alpha"])["alpha"].to_numpy()
            grid = np.load(gp).astype(float)
            mean_curve = np.load(cp).astype(np.float64).mean(axis=0)
            i_star = int(mean_curve.argmax())
            a_mean, a_med = float(a.mean()), float(np.median(a))
            i_mean = int(np.abs(grid - a_mean).argmin())
            i_med = int(np.abs(grid - a_med).argmin())
            rows.append(dict(dataset=ds, fusion=fu, split=s, n=len(a),
                             alpha_mean=a_mean, alpha_median=a_med,
                             alpha_star=float(grid[i_star]),
                             ndcg_star=float(mean_curve[i_star]),
                             ndcg_at_mean=float(mean_curve[i_mean]),
                             ndcg_at_median=float(mean_curve[i_med]),
                             penalty_mean=float(mean_curve[i_star] - mean_curve[i_mean]),
                             penalty_median=float(mean_curve[i_star] - mean_curve[i_med])))
    return pd.DataFrame(rows)


def fig_best_vs_mean_alpha(tab, out):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    mark = {"score-minmax": "o", "rrf": "s", "borda": "^"}
    col = {"score-minmax": BLUE, "rrf": GOLD, "borda": "#7b1fa2"}

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1.2, zorder=1)
    ax.annotate("if the best α were the mean α,\nevery point would sit on this line",
                (0.62, 0.55), fontsize=9, color="#444", rotation=0)
    for _, r in tab.iterrows():
        ax.scatter(r.alpha_mean, r.alpha_star, marker=mark[r.fusion], s=95,
                   color=col[r.fusion], edgecolor="white", lw=0.7, zorder=3)
        ax.plot([r.alpha_mean, r.alpha_mean], [r.alpha_mean, r.alpha_star],
                color=col[r.fusion], lw=0.9, alpha=0.45, zorder=2)
        _annotate(ax, r.alpha_mean, r.alpha_star, r.dataset, fs=8)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("mean of the per-query oracle α")
    ax.set_ylabel("α* that actually maximises mean NDCG@10")
    ax.set_title("The best global α is not the mean oracle α")
    ax.legend(handles=[Line2D([], [], marker=mark[f], ls="", color=col[f],
                              label=FLABEL[f]) for f in FUSIONS],
              loc="lower right", fontsize=9, framealpha=0.95, title="fusion")

    ax = axes[1]
    x = np.arange(len(DS_ORDER)); w = 0.26
    for j, fu in enumerate(FUSIONS):
        sub = tab[tab.fusion == fu].set_index("dataset").reindex(DS_ORDER)
        ax.bar(x + (j - 1) * w, sub.penalty_mean, w, color=col[fu],
               edgecolor="black", lw=0.5, label=FLABEL[fu])
    ax.set_xticks(x); ax.set_xticklabels(DS_ORDER, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("NDCG@10 lost by using mean α instead of α*")
    ax.set_title("Cost of the naive 'just average the oracle αs' shortcut")
    ax.legend(fontsize=8, framealpha=0.95, title="fusion")
    ax.axhline(0, color="black", lw=0.8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def build_all(paths=None, outdir=None):
    cfg = load_config()
    paths = paths or get_paths(cfg)
    outdir = outdir or os.path.join(paths["results"], "figures")
    os.makedirs(outdir, exist_ok=True)
    rf = paths["router_final"]
    summ = pd.read_csv(os.path.join(rf, "STUDY_SUMMARY.csv"))
    dev_ds = cfg["study"]["development_dataset"]

    made = {}
    p = os.path.join(outdir, "h1_combined.png")
    r = fig_h1_combined(summ, p); made["h1_combined"] = p
    p = os.path.join(outdir, "h1_by_fusion.png")
    rs = fig_h1_by_fusion(summ, p); made["h1_by_fusion"] = p
    p = os.path.join(outdir, "gain_by_fusion.png")
    fig_gain_by_fusion(summ, p); made["gain_by_fusion"] = p
    p = os.path.join(outdir, "headroom_by_fusion.png")
    fig_headroom_by_fusion(summ, p); made["headroom_by_fusion"] = p

    tab = alpha_star_table(paths)
    if len(tab):
        p = os.path.join(outdir, "best_vs_mean_alpha.png")
        fig_best_vs_mean_alpha(tab, p); made["best_vs_mean_alpha"] = p
        tab.round(4).to_csv(os.path.join(outdir, "best_vs_mean_alpha.csv"), index=False)

    h2p = os.path.join(rf, f"{dev_ds}_score-minmax_h2_decision_rule.csv")
    if os.path.exists(h2p):
        p = os.path.join(outdir, "h2.png")
        fig_h2(pd.read_csv(h2p), p); made["h2"] = p

    print(f"[plots] wrote {len(made)} figures -> {outdir}")
    if len(tab):
        print(f"[plots] best vs mean alpha: mean |a* - mean a| = "
              f"{(tab.alpha_star - tab.alpha_mean).abs().mean():.3f}, "
              f"mean NDCG penalty = {tab.penalty_mean.mean():.4f} "
              f"(worst {tab.penalty_mean.max():.4f})")
    print(f"[plots] H1 held-out correlation (all fusions pooled): r = {r:+.3f}")
    for fu, rr in rs.items():
        print(f"[plots]   {FLABEL[fu]:14s} r = {rr:+.3f}")
    return made


if __name__ == "__main__":
    build_all()
