md:# 90 — Comparison of all binary metaheuristics (UBF) with ANOVA-F and Chi-squared
# %%
import sys, warnings, json
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from fsbench import experiment as ex, plots
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
recs = ex.load_all()
df = ex.flatten(recs)
CLASSIC = ["AllFeatures", "ANOVA", "Chi2", "RS"]   # RS = random-search control under the same framework
print(df.method.nunique(), "methods,", len(recs), "runs")
# %%
md:## Main table: mean test metrics over 8 datasets x 6 classifiers x 10 runs
# %%
metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_macro", "n_selected", "reduction", "search_time_s"]
overall = df.groupby("method")[metrics].mean()
# rank per (dataset, classifier) block on mean F1, then average (Friedman ranks)
block = df.groupby(["dataset", "classifier", "method"]).f1_macro.mean().unstack()
ranks = block.rank(axis=1, ascending=False)
overall["mean_rank_f1"] = ranks.mean()
overall = overall.sort_values("mean_rank_f1")
overall.round(4).to_csv(TAB / "overall_ranking.csv")
overall.round(4)
# %%
chi, p = stats.friedmanchisquare(*[block[c].values for c in block.columns])
print(f"Friedman test over {block.shape[0]} blocks and {block.shape[1]} methods: chi2={chi:.2f}, p={p:.3e}")
meta = [m for m in overall.index if m not in CLASSIC]
FLAGSHIP = meta[0]          # best-ranked wrapper, used as the reference of the Wilcoxon tests below
TOP = meta[:3]              # reported as the top group (no single flagship: differences are not significant)
print("Top group (best mean ranks among the binary metaheuristics):", TOP)
json.dump({"flagship": FLAGSHIP, "top_group": TOP, "friedman_chi2": chi, "friedman_p": p}, open(TAB / "flagship.json", "w"))
# %%
md:## Wilcoxon signed-rank test: flagship vs every other method (paired over dataset x classifier blocks, Holm correction)
# %%
rows = []
for m in block.columns:
    if m == FLAGSHIP: continue
    d = block[FLAGSHIP] - block[m]
    stat, pv = stats.wilcoxon(block[FLAGSHIP], block[m], zero_method="zsplit") if np.any(d != 0) else (np.nan, 1.0)
    rows.append({"method": m, "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()), "p_value": pv})
w = pd.DataFrame(rows).sort_values("p_value")
k = len(w); w["p_holm"] = np.minimum(1, np.maximum.accumulate(w.p_value.values * (k - np.arange(k))))
w["significant_0.05"] = w.p_holm < 0.05
w.round(4).to_csv(TAB / "wilcoxon_flagship.csv", index=False)
w.round(4)
# %%
md:## Pairwise Wilcoxon tests inside the top group and against the classical baselines (Holm per family)
# %%
pairs = [(a, b) for i, a in enumerate(TOP) for b in TOP[i + 1:]] + [(a, b) for a in TOP for b in CLASSIC]
rows = []
for a, b in pairs:
    d = block[a] - block[b]
    pv = stats.wilcoxon(block[a], block[b], zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
    rows.append({"A": a, "B": b, "mean_F1_A": block[a].mean(), "mean_F1_B": block[b].mean(),
                 "wins_A": int((d > 0).sum()), "losses_A": int((d < 0).sum()), "p_value": pv})
pw = pd.DataFrame(rows).sort_values("p_value")
k = len(pw); pw["p_holm"] = np.minimum(1, np.maximum.accumulate(pw.p_value.values * (k - np.arange(k))))
pw.round(4).to_csv(TAB / "wilcoxon_top_group.csv", index=False); pw.round(4)
# %%
md:## Search quality: does each metaheuristic beat the random-search control?
Compared on the training-CV fitness (lower is better), paired over dataset x run with the same seeds.
# %%
fit = pd.DataFrame([{"method": r["method"], "dataset": r["dataset"], "run": r["run"], "fitness": r["best_fitness"],
                     "before_ls": r["history"][-2] if len(r.get("history", [])) > 1 else np.nan}
                    for r in recs if r["method"] not in ("AllFeatures", "ANOVA", "Chi2")])
piv = fit.pivot_table(index=["dataset", "run"], columns="method", values="before_ls")
rows = []
for m in piv.columns:
    if m == "RS": continue
    d = piv["RS"] - piv[m]                      # > 0: the metaheuristic found a better subset than RS
    pv = stats.wilcoxon(piv["RS"], piv[m], zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
    rows.append({"method": m, "mean_fitness_before_LS": piv[m].mean(), "RS_before_LS": piv["RS"].mean(),
                 "wins_vs_RS": int((d > 0).sum()), "losses_vs_RS": int((d < 0).sum()), "p_value": pv})
vs_rs = pd.DataFrame(rows).sort_values("mean_fitness_before_LS")
k = len(vs_rs); o = vs_rs.p_value.values.argsort(); ph = np.empty(k)
ph[o] = np.minimum(1, np.maximum.accumulate(vs_rs.p_value.values[o] * (k - np.arange(k))))
vs_rs["p_holm"] = ph
vs_rs["beats_RS"] = (vs_rs.p_holm < 0.05) & (vs_rs.mean_fitness_before_LS < vs_rs.RS_before_LS)
vs_rs.round(4).to_csv(TAB / "search_vs_random.csv", index=False)
print(int(vs_rs.beats_RS.sum()), "of", k, "metaheuristics beat the random-search control (Holm, 0.05)")
vs_rs.round(4)
# %%
md:## Per-dataset F1-macro (mean over classifiers and runs): top-10 metaheuristics vs classical filters
# %%
top = meta[:10] + CLASSIC
per_ds = df[df.method.isin(top)].groupby(["dataset", "method"]).f1_macro.mean().unstack()[top].round(4)
per_ds.to_csv(TAB / "f1_per_dataset_top10.csv"); per_ds
# %%
for metric in ["recall_macro", "precision_macro", "auc_macro", "accuracy"]:
    t = df[df.method.isin(top)].groupby(["dataset", "method"])[metric].mean().unstack()[top].round(4)
    t.to_csv(TAB / f"{metric}_per_dataset_top10.csv")
    display(t.style.set_caption(metric).highlight_max(axis=1))
# %%
md:## Per-classifier F1-macro (top group vs classical baselines)
# %%
pc = df[df.method.isin(TOP + CLASSIC)].groupby(["classifier", "method"]).f1_macro.mean().unstack().round(4)
pc.to_csv(TAB / "f1_per_classifier_flagship.csv"); pc
# %%
md:## Computation: search time, real classifier trainings and memoisation hits
# %%
comp = df.drop_duplicates(["method", "dataset", "run"]).groupby("method")[["search_time_s", "n_classifier_fits", "cache_hits", "n_selected"]].mean()
comp["cache_saving_%"] = 100 * comp.cache_hits / (comp.cache_hits + comp.n_classifier_fits).replace(0, np.nan)
comp = comp.loc[overall.index].round(2); comp.to_csv(TAB / "computation.csv"); comp
# %%
md:## Figures: F1 distribution, convergence, ROC and confusion matrices
# %%
fig, ax = plt.subplots(figsize=(14, 4))
order = list(overall.index)
ax.boxplot([df[df.method == m].f1_macro for m in order], labels=order, showfliers=False)
ax.set_ylabel("test F1-macro"); plt.xticks(rotation=90); plt.tight_layout()
fig.savefig(ROOT / "results/figures/boxplot_f1_all.png", dpi=200)
# %%
by = {m: [r for r in recs if r["method"] == m] for m in df.method.unique()}
for ds in ex.DATASETS:
    plots.convergence_compare({m: by[m] for m in meta[:8]}, ds, name=f"convergence_top8_{ds}")
# %%
sel = {m: by[m] for m in TOP + CLASSIC}
for ds in ["wdbc", "ionosphere", "sonar", "spectf"]:
    plots.roc_curves(sel, ds, classifier="KNN", name=f"roc_flagship_{ds}")
# %%
for ds in ["vehicle", "dermatology", "wine"]:
    plots.confusion_matrices(sel, ds, classifier="KNN", name=f"cm_flagship_{ds}")
