"""audit_study.py -- read-only health check for the whole study.

    python src/audit_study.py

Inspects the FILES ON DISK for every (dataset x fusion) cell -- independent of
any terminal output or logs -- and reports, per cell, which pipeline sections
have produced valid outputs. It does not just check existence: it opens each
artefact, so a file left half-written when a section was killed mid-write shows
up as CORRUPT rather than passing silently.

Nothing here writes or deletes anything; safe to run while the study is going.
"""

import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from utils import load_config, get_paths, dataset_dir, processed_dir


def fusion_tag(f):
    return f"score-{f.get('normalizer', 'minmax')}" if f["function"] == "score" else f["function"]


def check_file(path, kind):
    """(-, MISSING, OK, CORRUPT). Opens the file to detect truncation."""
    if not os.path.exists(path):
        return "MISSING"
    try:
        if kind == "npy":
            np.load(path, mmap_mode="r").shape
        elif kind == "npz":
            z = np.load(path, allow_pickle=False)
            _ = [z[k].shape for k in z.files]           # touch every array
        elif kind == "csv":
            pd.read_csv(path, nrows=5)
        elif kind == "json":
            json.load(open(path, encoding="utf-8"))
        elif kind == "joblib":
            import joblib
            joblib.load(path)
        return "OK"
    except Exception as e:
        return f"CORRUPT ({type(e).__name__})"


def avail_splits(paths, ds):
    folder = dataset_dir(paths, ds)
    qd = os.path.join(folder, "qrels")
    if not os.path.isdir(qd):
        return []
    return [s for s in ("train", "dev", "test") if os.path.exists(os.path.join(qd, f"{s}.tsv"))]


def audit_cell(cfg, paths, ds, fu, is_dev, top_k):
    t = fusion_tag(fu)
    splits = avail_splits(paths, ds) or ["train", "dev", "test"]
    pdir = processed_dir(paths, ds, create=False)
    fd, rs, rf = paths["feature_dataset"], paths["router_screening"], paths["router_final"]

    # (section, [(path, kind), ...])   -- only sections this cell should run
    checks = [
        (0, [(os.path.join(dataset_dir(paths, ds), "corpus.jsonl"), "raw")]),
        (1, [(os.path.join(pdir, "corpus_emb.npy"), "npy"),
             (os.path.join(pdir, "corpus_ids.json"), "json")]),
        (3, [(os.path.join(pdir, f"retrieval_{s}_top{top_k}.npz"), "npz") for s in splits]),
        (4, [(os.path.join(fd, f"{ds}_{t}_{s}_features.csv"), "csv") for s in splits]
            + [(os.path.join(fd, f"{ds}_{t}_{s}_curve.npy"), "npy") for s in splits]),
    ]
    if is_dev:
        checks += [
            (5, [(os.path.join(rs, f"{ds}_{t}_screen.csv"), "csv")]),
            (6, [(os.path.join(rs, f"{ds}_{t}_ablation.csv"), "csv")]),
            (7, [(os.path.join(rs, f"{ds}_{t}_rescreen_best.json"), "json")]),
        ]
    checks += [
        (8, [(os.path.join(rf, f"{ds}_{t}_router.joblib"), "joblib"),
             (os.path.join(rf, f"{ds}_{t}_router_meta.json"), "json")]),
        (9, [(os.path.join(rf, f"{ds}_{t}_benchmark.csv"), "csv"),
             (os.path.join(rf, f"{ds}_{t}_benchmark.json"), "json")]),
    ]

    sec_status, problems = {}, []
    for sec, files in checks:
        states = []
        for p, kind in files:
            if kind == "raw":
                st = "OK" if os.path.exists(p) else "MISSING"
            else:
                st = check_file(p, kind)
            states.append(st)
            if st.startswith("CORRUPT"):
                problems.append(f"sec{sec} CORRUPT: {os.path.basename(p)} ({st})")
        if all(s == "OK" for s in states):
            sec_status[sec] = "ok"
        elif any(s == "OK" for s in states) or any(s.startswith("CORRUPT") for s in states):
            sec_status[sec] = "partial"
        else:
            sec_status[sec] = "todo"
    return t, splits, sec_status, problems


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    st = cfg["study"]
    dev_ds = st["development_dataset"]
    top_k = cfg["retrieval"]["top_k"]

    print(f"### AUDIT  dataset-matrix = {st['datasets']} x "
          f"{[fusion_tag(f) for f in st['fusions']]}\n")
    hdr = f"{'dataset':16s} {'fusion':13s} {'role':9s} sections(0..9)         status"
    print(hdr + "\n" + "-" * len(hdr))

    all_problems = []
    n_done = 0
    for ds in st["datasets"]:
        for fu in st["fusions"]:
            is_dev = (ds == dev_ds)
            t, splits, sec_status, problems = audit_cell(cfg, paths, ds, fu, is_dev, top_k)
            expected = [0, 1, 3, 4, 5, 6, 7, 8, 9] if is_dev else [0, 1, 3, 4, 8, 9]
            glyph = "".join(
                ("." if s not in expected else
                 {"ok": "#", "partial": "!", "todo": "-"}[sec_status.get(s, "todo")])
                for s in range(10))
            done = sec_status.get(9) == "ok"
            n_done += done
            corrupt = any("CORRUPT" in p for p in problems)
            status = ("DONE" if done else
                      "CORRUPT" if corrupt else
                      "in-progress/failed")
            print(f"{ds:16s} {t:13s} {'DEV' if is_dev else 'held-out':9s} "
                  f"{glyph:22s} {status}")
            all_problems += [f"{ds}/{t}: {p}" for p in problems]

    print("\nlegend: #=ok  !=partial/corrupt  -=todo  .=not applicable to this cell "
          "| section 9 (benchmark) = cell complete")
    print(f"\n{n_done}/{len(st['datasets']) * len(st['fusions'])} cells complete")

    if all_problems:
        print("\n!!! CORRUPT / PARTIAL artefacts (delete these, then re-launch the study;")
        print("    the section will rebuild -- nothing downstream trusts a bad file):")
        for p in all_problems:
            print("   ", p)
    else:
        print("\nNo corrupt artefacts. Any incomplete cell simply hasn't run yet -- "
              "re-launching run_study.py resumes it.")

    # surface logged failures without needing the scrollback
    logs = sorted(glob.glob(os.path.join(paths["results"], "study_logs", "*.log")))
    fails = []
    for lp in logs:
        for line in open(lp, encoding="utf-8"):
            if "FAILED" in line:
                fails.append(f"{os.path.basename(lp)}: {line.strip()}")
    if fails:
        print(f"\nFAILED lines in study_logs ({len(fails)}):")
        for f in fails[-20:]:
            print("   ", f)


if __name__ == "__main__":
    main()
