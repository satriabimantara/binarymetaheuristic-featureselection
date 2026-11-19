"""Generate the experiment notebooks (one per binary metaheuristic + protocol, baselines, comparison, ablation)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
SETUP = """import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fsbench import experiment as ex, plots
pd.set_option("display.width", 200, "display.max_columns", 40)"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n


def algorithm_notebook(i, name, full, ref):
    return nb([
        f"md:# {i:02d} — Binary {full} (B{name}) for feature selection\n\n"
        f"Continuous algorithm: *{ref}*  \nBinarised with the **Unified Binarization Framework (UBF)**: deterministic sign mapping, "
        "density-diverse initial population, fitness memoisation and elite bit-flip local search. "
        "RS = random-search control under the same framework. Protocol: see `00_data_and_protocol`.",
        SETUP,
        f'METHOD = "{name}"\nrecs = ex.run_method(METHOD)\nbase = {{m: ex.run_method(m) for m in ("ANOVA", "Chi2", "AllFeatures", "RS")}}\n'
        'df = ex.flatten(recs)\nprint(len(recs), "runs")',
        "md:## Performance on the test split (mean ± std over 10 runs)",
        'plots.summary_table(df, by=("dataset", "classifier"))',
        "md:## Comparison with the classical filters (F1-macro, averaged over the six classifiers)",
        'allb = pd.concat([df] + [ex.flatten(r) for r in base.values()])\n'
        'pivot = allb.groupby(["dataset", "method"]).f1_macro.mean().unstack().round(4)\npivot',
        'allb.groupby(["dataset", "method"])[["n_selected", "search_time_s"]].mean().unstack().round(2)',
        "md:## Convergence: best-so-far fitness per iteration\nA flat tail means the search converged within the budget; "
        "a still-decreasing tail means more iterations could help.",
        f'plots.convergence(recs, title="B{name} convergence (mean and min-max over runs)", name="convergence_{name}");',
        "md:## Confusion matrices and ROC curves (run 0, KNN)",
        'ds = "vehicle"\nplots.confusion_matrices({METHOD: recs, **base}, ds, name=f"cm_{METHOD}_{ds}");',
        'plots.roc_curves({METHOD: recs, **base}, "wdbc", name=f"roc_{METHOD}_wdbc");',
        "md:## Computation\n`n_classifier_fits` counts real classifier trainings; `cache_hits` are FEs served by the memo.",
        'pd.DataFrame([{k: r[k] for k in ("dataset", "run", "n_selected", "time_s", "n_classifier_fits", "cache_hits")} '
        'for r in recs]).groupby("dataset").mean().round(2)',
    ])


def main():
    from metaheuristicpy.core import ALGORITHMS

    NB.mkdir(exist_ok=True)
    nbf.write(nb([
        "md:# 00 — Datasets and experimental protocol",
        SETUP,
        "ex.dataset_table()",
        "print(ex.__doc__)\nprint('search settings:', ex.SEARCH, '| runs:', ex.N_RUNS)\nprint('classifiers:', list(ex.CLASSIFIERS))",
        "from metaheuristicpy.core import ALGORITHMS\npd.DataFrame([{'acronym': k, 'algorithm': v.full_name, "
        "'reference': v.reference} for k, v in ALGORITHMS.items()])",
    ]), NB / "00_data_and_protocol.ipynb")
    nbf.write(nb([
        "md:# 01 — Classical feature selection baselines (all features, ANOVA F, Chi-squared)",
        SETUP,
        'recs = {m: ex.run_method(m) for m in ("AllFeatures", "ANOVA", "Chi2")}\n'
        'df = pd.concat([ex.flatten(r) for r in recs.values()])',
        'plots.summary_table(df, by=("dataset", "method", "classifier"))',
        'df.groupby(["dataset", "method"]).n_selected.first().unstack()',
    ]), NB / "01_baselines_classic.ipynb")
    for i, (name, cls) in enumerate(ALGORITHMS.items(), start=2):
        nbf.write(algorithm_notebook(i, name, cls.full_name, cls.reference), NB / f"{i:02d}_B{name}.ipynb")
    nbf.write(nb(open(ROOT / "scripts/comparison_cells.py", encoding="utf-8").read().split("\n# %%\n")),
              NB / "90_comparison_all_methods.ipynb")
    nbf.write(nb(open(ROOT / "scripts/ablation_cells.py", encoding="utf-8").read().split("\n# %%\n")),
              NB / "91_ablation_UBF.ipynb")
    print("notebooks written to", NB)


if __name__ == "__main__":
    main()
