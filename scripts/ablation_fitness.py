"""Ablation on the search itself: training-CV fitness of every variant against the full framework, paired over
dataset x run (the notebook 91 table reports the test split only).
Writes results/tables/ablation_fitness.csv.   python scripts/ablation_fitness.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW, TAB = ROOT / "results/raw", ROOT / "results/tables"


def load(path):
    return {(r["dataset"], r["run"]): r for r in pickle.load(open(path, "rb"))}


rows = []
for alg in json.load(open(TAB / "flagship.json"))["top_group"][:2]:
    full = load(RAW / f"{alg}.pkl")
    keys = sorted(full)
    for path in sorted((RAW / "ablation").glob(f"{alg}__*.pkl")):
        var = load(path)
        for stage, get in (("before_local_search", lambda r: r["history"][-2]), ("final", lambda r: r["best_fitness"])):
            f, v = np.array([get(full[k]) for k in keys]), np.array([get(var[k]) for k in keys])
            p = stats.wilcoxon(f, v, zero_method="zsplit").pvalue if np.any(f != v) else 1.0
            rows.append({"algorithm": alg, "variant": path.stem.split("__")[1], "stage": stage, "fitness_full": f.mean(),
                         "fitness_variant": v.mean(), "full_wins": int((f < v).sum()), "full_losses": int((f > v).sum()),
                         "p_wilcoxon": p})
pd.DataFrame(rows).round(4).to_csv(TAB / "ablation_fitness.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())
