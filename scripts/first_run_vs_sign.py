"""First run (stochastic time-varying sigmoid, uniform initialisation) against the reported run (sign mapping,
density-diverse initialisation) and against the random-search control RS.

Paired over dataset x run on the training-CV fitness reached before the elite local search (lower is better).
musk1 is left out: in the first run it still contained the two identifier columns.
Writes results/tables/stochastic_vs_sign.csv and stochastic_vs_sign.json.   python scripts/first_run_vs_sign.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OLD, NEW, TAB = ROOT / "results_v1_stochastic_ubf/raw", ROOT / "results/raw", ROOT / "results/tables"


def before_ls(path):
    return {(r["dataset"], r["run"]): r["history"][-2] for r in pickle.load(open(path, "rb"))
            if len(r.get("history", [])) > 1 and r["dataset"] != "musk1"}


def holm(p):
    p = np.asarray(p); o = p.argsort(); out = np.empty(len(p))
    out[o] = np.minimum(1, np.maximum.accumulate(p[o] * (len(p) - np.arange(len(p)))))
    return out


rs = before_ls(NEW / "RS.pkl")
rows = []
for path in sorted(OLD.glob("*.pkl")):
    old = before_ls(path)
    if not old:
        continue                                             # AllFeatures, ANOVA, Chi2 have no search history
    new = before_ls(NEW / path.name)
    keys = sorted(set(old) & set(new) & set(rs))
    o, n, r = (np.array([d[k] for k in keys]) for d in (old, new, rs))
    w = lambda a, b: stats.wilcoxon(a, b, zero_method="zsplit").pvalue  # noqa: E731
    rows.append({"method": path.stem, "pairs": len(keys), "fitness_stochastic": o.mean(), "fitness_sign": n.mean(),
                 "fitness_RS": r.mean(), "p_stochastic_vs_RS": w(o, r), "p_sign_vs_RS": w(n, r), "p_sign_vs_stochastic": w(n, o),
                 "sign_wins": int((n < o).sum()), "sign_losses": int((n > o).sum())})
d = pd.DataFrame(rows)
for a in ("stochastic_vs_RS", "sign_vs_RS", "sign_vs_stochastic"):
    d[f"p_holm_{a}"] = holm(d[f"p_{a}"])
d["stochastic_beats_RS"] = (d.p_holm_stochastic_vs_RS < 0.05) & (d.fitness_stochastic < d.fitness_RS)
d["sign_beats_RS"] = (d.p_holm_sign_vs_RS < 0.05) & (d.fitness_sign < d.fitness_RS)
d["sign_worse_than_RS"] = (d.p_holm_sign_vs_RS < 0.05) & (d.fitness_sign > d.fitness_RS)
d["sign_better"] = (d.p_holm_sign_vs_stochastic < 0.05) & (d.fitness_sign < d.fitness_stochastic)
d["sign_worse"] = (d.p_holm_sign_vs_stochastic < 0.05) & (d.fitness_sign > d.fitness_stochastic)
d = d.sort_values("fitness_sign")
d.round(4).to_csv(TAB / "stochastic_vs_sign.csv", index=False)
summary = {"n_algorithms": len(d), "pairs": int(d.pairs.iloc[0]),
           "mean_fitness_stochastic": float(d.fitness_stochastic.mean()), "mean_fitness_sign": float(d.fitness_sign.mean()),
           "fitness_RS": float(d.fitness_RS.mean()),
           "stochastic_beats_RS": int(d.stochastic_beats_RS.sum()), "sign_beats_RS": int(d.sign_beats_RS.sum()),
           "sign_worse_than_RS": sorted(d.method[d.sign_worse_than_RS]),
           "sign_better": int(d.sign_better.sum()), "sign_worse": sorted(d.method[d.sign_worse])}
json.dump(summary, open(TAB / "stochastic_vs_sign.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
