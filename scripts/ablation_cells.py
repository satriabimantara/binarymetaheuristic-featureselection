md:# 91 — Ablation of the Unified Binarization Framework components (two best-ranked algorithms)
# %%
import sys, warnings, json
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from fsbench import experiment as ex, plots
ALGS = json.load(open(ROOT / "results/tables/flagship.json"))["top_group"][:2]
print("ablated algorithms:", ALGS)
# %%
variants = {
    "UBF (sign mapping + diverse init + memo + local search)": dict(),
    "stochastic TV-sigmoid instead of sign mapping": dict(transfer="tvs"),
    "stochastic S2 sigmoid instead of sign mapping": dict(transfer="s2"),
    "uniform instead of density-diverse init": dict(init="uniform"),
    "no elite local search": dict(ls_ratio=0.0),
    "no memoisation": dict(cache=False),
}
res = {}
for alg in ALGS:
    for label, kw in variants.items():
        tag = alg if not kw else f"{alg}__" + "_".join(f"{k}-{v}" for k, v in kw.items())
        recs = ex.run_method(alg, tag=tag, subdir=None if not kw else "ablation", **kw)
        recs = [dict(r, method=label, algorithm=alg) for r in recs]
        res[(alg, label)] = recs
df = pd.concat([ex.flatten(r).assign(algorithm=alg) for (alg, _), r in res.items()])
# %%
t = df.groupby(["algorithm", "method"])[["f1_macro", "accuracy", "auc_macro", "n_selected", "search_time_s", "n_classifier_fits", "cache_hits"]].mean().round(4)
t.to_csv(ROOT / "results/tables/ablation.csv"); t
# %%
full = list(variants)[0]
out = []
for alg in ALGS:
    blk = df[df.algorithm == alg].groupby(["dataset", "classifier", "method"]).f1_macro.mean().unstack()
    for m in blk.columns:
        out.append({"algorithm": alg, "variant": m, "delta_f1_vs_full": (blk[m] - blk[full]).mean(),
                    "p_wilcoxon": stats.wilcoxon(blk[full], blk[m], zero_method="zsplit").pvalue if m != full else np.nan})
pd.DataFrame(out).round(4).to_csv(ROOT / "results/tables/ablation_wilcoxon.csv", index=False); pd.DataFrame(out).round(4)
# %%
for alg in ALGS:
    plots.convergence_compare({lab: r for (a, lab), r in res.items() if a == alg}, "sonar", name=f"ablation_convergence_{alg}_sonar")
