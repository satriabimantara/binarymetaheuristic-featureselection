"""Figures used by the notebooks: convergence (fitness per iteration), ROC curves, confusion matrices."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def _save(fig, name):
    if name:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


def convergence(records, title="", name=None, ncols=4):
    """Fitness of the best-so-far subset per iteration: mean (line) and min-max band over runs, per dataset."""
    datasets = list(dict.fromkeys(r["dataset"] for r in records))
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, ds in zip(axes.flat, datasets):
        curves = [r["history"] for r in records if r["dataset"] == ds and r["history"]]
        L = min(len(c) for c in curves)
        C = np.array([c[:L] for c in curves])
        x = np.arange(1, L + 1)
        ax.plot(x, C.mean(0), lw=2)
        ax.fill_between(x, C.min(0), C.max(0), alpha=0.25)
        ax.set_title(ds)
        ax.set_xlabel("iteration")
        ax.set_ylabel("fitness (lower is better)")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(datasets):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, name)
    return fig


def convergence_compare(records_by_method, dataset, name=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, recs in records_by_method.items():
        curves = [r["history"] for r in recs if r["dataset"] == dataset and r["history"]]
        if not curves:
            continue
        L = min(len(c) for c in curves)
        ax.plot(np.arange(1, L + 1), np.mean([c[:L] for c in curves], 0), label=method)
    ax.set(title=f"Mean convergence on {dataset}", xlabel="iteration", ylabel="fitness")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, name)
    return fig


def roc_curves(records_by_method, dataset, classifier="KNN", name=None):
    """ROC of run 0 of each method (micro-averaged for multi-class)."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for method, recs in records_by_method.items():
        r = next((r for r in recs if r["dataset"] == dataset and r["run"] == 0), None)
        if r is None:
            continue
        fpr, tpr = r["test"][classifier]["roc"]
        ax.plot(fpr, tpr, label=f"{method} (AUC={r['test'][classifier]['auc_macro']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(title=f"ROC - {dataset} - {classifier}", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(fontsize=7)
    _save(fig, name)
    return fig


def confusion_matrices(records_by_method, dataset, classifier="KNN", name=None):
    methods = list(records_by_method)
    fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 3), squeeze=False)
    for ax, m in zip(axes.flat, methods):
        r = next(r for r in records_by_method[m] if r["dataset"] == dataset and r["run"] == 0)
        cm = r["test"][classifier]["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", fontsize=8,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set(title=m, xlabel="predicted", ylabel="true")
    fig.suptitle(f"Confusion matrices - {dataset} - {classifier}")
    fig.tight_layout()
    _save(fig, name)
    return fig


def summary_table(df, by=("dataset",), metric_cols=("accuracy", "precision_macro", "recall_macro", "f1_macro",
                                                    "auc_macro", "n_selected", "search_time_s")):
    g = df.groupby(list(by))[list(metric_cols)]
    out = g.mean().round(4).astype(str) + " ± " + g.std().fillna(0).round(4).astype(str)
    return out.reset_index() if isinstance(out, pd.DataFrame) else out
