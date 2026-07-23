"""Boxplots of the oracle alpha distribution per dataset.

    python src/plot_alpha.py                          # all datasets x fusions
    python src/plot_alpha.py --split test             # test-split queries only
    python src/plot_alpha.py --fusion score-minmax    # one fusion only

Reads the oracle alpha label from the section-4 feature CSVs. Alpha is the
per-query oracle fusion weight (1 = pure lexical, 0 = pure dense); its IQR is the
per-query routing headroom that H1 tests against.

Outputs (data/results/alpha_distribution/):
    oracle_alpha_summary.csv         n / median / mean / std / IQR per (ds, fusion)
    oracle_alpha_boxplot_<tag>.png   one box per dataset, per fusion
    oracle_alpha_boxplot_grouped.png datasets x fusions on one axis
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_config, get_paths


def fusion_tag(f):
    return f"score-{f.get('normalizer', 'minmax')}" if f["function"] == "score" else f["function"]


def load_alpha(fd, ds, tag, split):
    """Concatenate the `alpha` column over the requested split(s)."""
    if split == "all":
        paths = sorted(glob.glob(os.path.join(fd, f"{ds}_{tag}_*_features.csv")))
    else:
        p = os.path.join(fd, f"{ds}_{tag}_{split}_features.csv")
        paths = [p] if os.path.exists(p) else []
    vals = []
    for p in paths:
        try:
            vals.append(pd.read_csv(p, usecols=["alpha"])["alpha"].to_numpy())
        except Exception:
            pass
    return np.concatenate(vals) if vals else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["all", "train", "dev", "test"])
    ap.add_argument("--fusion", default=None, help="restrict to one fusion tag")
    args = ap.parse_args()

    cfg = load_config()
    paths = get_paths(cfg)
    fd = paths["feature_dataset"]
    outdir = paths.get("alpha_results") or os.path.join(paths["results"], "alpha_distribution")
    os.makedirs(outdir, exist_ok=True)

    datasets = cfg["study"]["datasets"]                  # ordered by alpha spread
    tags = [fusion_tag(f) for f in cfg["study"]["fusions"]]
    if args.fusion:
        tags = [args.fusion]

    # ---- collect ----
    data = {}                                            # (ds, tag) -> alpha array
    recs = []
    for ds in datasets:
        for tag in tags:
            a = load_alpha(fd, ds, tag, args.split)
            if a.size == 0:
                continue
            data[(ds, tag)] = a
            q1, q3 = np.percentile(a, [25, 75])
            recs.append(dict(dataset=ds, fusion=tag, n=len(a),
                             alpha_mean=round(float(a.mean()), 4),
                             alpha_median=round(float(np.median(a)), 4),
                             alpha_std=round(float(a.std()), 4),
                             alpha_iqr=round(float(q3 - q1), 4),
                             frac_lexical=round(float((a > 0.9).mean()), 3),
                             frac_dense=round(float((a < 0.1).mean()), 3)))
    if not recs:
        raise SystemExit(f"[plot] no feature CSVs found in {fd} for split='{args.split}'. "
                         f"Run the pipeline (section 4) first.")
    summ = pd.DataFrame(recs)
    scsv = os.path.join(outdir, "oracle_alpha_summary.csv")
    summ.to_csv(scsv, index=False)
    print(f"[plot] wrote {scsv}\n")
    print(summ.to_string(index=False))

    present_ds = [d for d in datasets if any((d, t) in data for t in tags)]

    # ---- one combined boxplot per fusion (datasets kept in spread order) ----
    for tag in tags:
        series, labels = [], []
        for ds in present_ds:
            if (ds, tag) in data:
                series.append(data[(ds, tag)])
                iqr = summ[(summ.dataset == ds) & (summ.fusion == tag)]["alpha_iqr"].iloc[0]
                labels.append(f"{ds}\nIQR={iqr:.2f}")
        if not series:
            continue
        plt.figure(figsize=(max(7, 1.5 * len(series)), 5))
        plt.boxplot(series, labels=labels, showmeans=True, widths=0.55,
                    medianprops=dict(color="navy"),
                    meanprops=dict(marker="D", markerfacecolor="orange", markersize=5))
        plt.axhline(0.5, ls="--", color="red", lw=1)
        plt.ylim(-0.03, 1.03)
        plt.ylabel("oracle alpha   (1 = BM25 / lexical,  0 = dense / semantic)")
        plt.title(f"Oracle alpha per dataset -- fusion={tag}, split={args.split}")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        out = os.path.join(outdir, f"oracle_alpha_boxplot_{tag}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"\n[plot] wrote {out}")

    # ---- grouped: datasets x fusions on one axis ----
    if len(tags) > 1:
        colors = plt.cm.Set2(np.linspace(0, 1, len(tags)))
        width = 0.8 / len(tags)
        plt.figure(figsize=(max(8, 1.8 * len(present_ds)), 5.5))
        for j, tag in enumerate(tags):
            positions, series = [], []
            for i, ds in enumerate(present_ds):
                if (ds, tag) in data:
                    positions.append(i + (j - (len(tags) - 1) / 2) * width)
                    series.append(data[(ds, tag)])
            if not series:
                continue
            bp = plt.boxplot(series, positions=positions, widths=width * 0.9,
                             patch_artist=True, showfliers=False,
                             medianprops=dict(color="black"))
            for box in bp["boxes"]:
                box.set(facecolor=colors[j], alpha=0.75)
            plt.plot([], [], color=colors[j], lw=6, label=tag)      # legend proxy
        plt.axhline(0.5, ls="--", color="red", lw=1)
        plt.ylim(-0.03, 1.03)
        plt.xticks(range(len(present_ds)), present_ds, rotation=20, ha="right")
        plt.ylabel("oracle alpha   (1 = BM25 / lexical,  0 = dense / semantic)")
        plt.title(f"Oracle alpha per dataset x fusion (split={args.split})")
        plt.legend(title="fusion", loc="upper right")
        plt.tight_layout()
        out = os.path.join(outdir, "oracle_alpha_boxplot_grouped.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
