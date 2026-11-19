"""Shared experiment protocol for the binary-metaheuristic feature-selection study.

Protocol (identical for every method, see notebooks/00_protocol.ipynb):
* 8 UCI datasets, stratified 70/30 train/test split (seed 2025), min-max scaling fitted on train only.
* Wrapper search on the TRAIN split only: fitness = 0.99 * (1 - F1-macro of 3-fold CV KNN(k=5))
  + 0.01 * |S|/D, n_agents = 20, 50 iterations (1000 FEs incl. 10% elite local search), 10 independent runs.
* The selected subset is then evaluated on the untouched TEST split with six classifiers
  (KNN, DT, SVM, ANN, RF, XGBoost): accuracy, macro precision/recall/F1, ROC-AUC (macro OvR), confusion matrix.
* Classical baselines: all features, SelectKBest(ANOVA f_classif) and SelectKBest(chi2), k chosen by the same
  3-fold CV on train over k in {10%, 20%, ..., 100%} of D.
* Control: RS = random search under the same framework (same density-diverse sampling, FE budget, memo and
  elite local search). A binary metaheuristic that does not beat RS is not contributing a search mechanism.
* UBF defaults: deterministic sign mapping (feature selected when its coordinate is positive) and
  density-diverse initial population. musk1: the molecule/conformation identifier columns are removed.
Raw results are cached in results/raw/<METHOD>.pkl so that every notebook can be re-executed cheaply.
"""
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, RAW_DIR = ROOT / "datasets", ROOT / "results" / "raw"
DATASETS = ["wine", "wdbc", "ionosphere", "sonar", "vehicle", "dermatology", "spectf", "musk1"]
SPLIT_SEED, N_RUNS = 2025, 10
SEARCH = {"n_agents": 20, "max_iter": 50, "alpha": 0.99, "cv": 3, "metric": "f1_macro", "ls_ratio": 0.1}
CLASSIFIERS = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DT": DecisionTreeClassifier(random_state=0),
    "SVM": SVC(kernel="rbf", probability=True, random_state=0),
    "ANN": MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=0),
    "RF": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1),
    "XGB": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=0, n_jobs=1,
                         eval_metric="mlogloss", verbosity=0),
}


def load_dataset(name):
    """Return (X_train, X_test, y_train, y_test, feature_names) with leakage-free scaling."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    y = LabelEncoder().fit_transform(df.pop("target").astype(str))
    df = df.loc[:, df.nunique() > 1].astype(float)
    df = df.fillna(df.median())
    Xtr, Xte, ytr, yte = train_test_split(df.values, y, test_size=0.3, stratify=y, random_state=SPLIT_SEED)
    sc = MinMaxScaler().fit(Xtr)
    return np.clip(sc.transform(Xtr), 0, 1), np.clip(sc.transform(Xte), 0, 1), ytr, yte, list(df.columns)


def dataset_table():
    rows = []
    for n in DATASETS:
        Xtr, Xte, ytr, yte, cols = load_dataset(n)
        rows.append({"dataset": n, "instances": len(ytr) + len(yte), "features": len(cols),
                     "classes": len(np.unique(ytr)), "train": len(ytr), "test": len(yte)})
    return pd.DataFrame(rows)


def evaluate_subset(Xtr, Xte, ytr, yte, cols):
    """Train the six classifiers on the selected columns and score them on the test split."""
    out = {}
    classes = np.unique(np.concatenate([ytr, yte]))
    for name, clf in CLASSIFIERS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = time.perf_counter()
            m = clone(clf).fit(Xtr[:, cols], ytr)
            pred = m.predict(Xte[:, cols])
            proba = m.predict_proba(Xte[:, cols])
            fit_time = time.perf_counter() - t
        if len(classes) == 2:
            auc = roc_auc_score(yte, proba[:, 1])
            fpr, tpr, _ = roc_curve(yte, proba[:, 1])
        else:
            auc = roc_auc_score(label_binarize(yte, classes=classes), proba, average="macro", multi_class="ovr")
            Yb = label_binarize(yte, classes=classes)
            fpr, tpr, _ = roc_curve(Yb.ravel(), proba.ravel())             # micro-averaged curve for plotting
        out[name] = {
            "accuracy": accuracy_score(yte, pred),
            "precision_macro": precision_score(yte, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(yte, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
            "auc_macro": auc, "confusion_matrix": confusion_matrix(yte, pred, labels=classes),
            "roc": (fpr, tpr), "fit_time_s": fit_time,
        }
    return out


def _search_run(method, dataset, run, transfer="sign", **overrides):
    from metaheuristicpy.discrete.feature_selection import UnifiedBinaryFS

    Xtr, Xte, ytr, yte, cols = load_dataset(dataset)
    cfg = {**SEARCH, **overrides}
    fs = UnifiedBinaryFS(method, transfer=transfer, seed=1000 * run + 17, **cfg).fit(Xtr, ytr)
    rec = {"method": method, "dataset": dataset, "run": run, "transfer": transfer, **fs.summary(),
           "history": list(fs.history_), "n_total_features": len(cols)}
    rec["test"] = evaluate_subset(Xtr, Xte, ytr, yte, fs.selected_features_)
    return rec


def _classic_run(method, dataset):
    Xtr, Xte, ytr, yte, cols = load_dataset(dataset)
    D = len(cols)
    t = time.perf_counter()
    if method == "AllFeatures":
        sel = np.arange(D)
    else:
        score = {"ANOVA": f_classif, "Chi2": chi2}[method]
        grid = sorted({max(1, int(round(D * q / 10))) for q in range(1, 11)})
        cv = StratifiedKFold(3, shuffle=True, random_state=SPLIT_SEED)
        best_k, best = grid[-1], -1
        for k in grid:
            cols_k = SelectKBest(score, k=k).fit(Xtr, ytr).get_support(indices=True)
            s = cross_val_score(KNeighborsClassifier(5), Xtr[:, cols_k], ytr, cv=cv, scoring="f1_macro").mean()
            if s > best:
                best_k, best = k, s
        sel = SelectKBest(score, k=best_k).fit(Xtr, ytr).get_support(indices=True)
    rec = {"method": method, "dataset": dataset, "run": 0, "selected_features": sel.tolist(),
           "n_selected": int(len(sel)), "n_total_features": D, "time_s": time.perf_counter() - t,
           "history": [], "n_evaluations": 0, "n_classifier_fits": 0, "cache_hits": 0}
    rec["test"] = evaluate_subset(Xtr, Xte, ytr, yte, sel)
    return rec


def run_method(method, datasets=DATASETS, runs=N_RUNS, n_jobs=None, force=False, tag=None, subdir=None, **kwargs):
    """Run (or load from cache) one method on every dataset; returns the list of run records.
    ``tag``/``subdir`` store variants (e.g. ablations) apart from the main results."""
    folder = RAW_DIR / subdir if subdir else RAW_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{tag or method}.pkl"
    if path.exists() and not force:
        return pickle.load(open(path, "rb"))
    n_jobs = n_jobs or int(os.environ.get("FSBENCH_JOBS", max(1, (os.cpu_count() or 2) - 2)))
    if method in ("AllFeatures", "ANOVA", "Chi2"):
        tasks = [delayed(_classic_run)(method, d) for d in datasets]
    else:
        tasks = [delayed(_search_run)(method, d, r, **kwargs) for d in datasets for r in range(runs)]
    recs = Parallel(n_jobs=n_jobs)(tasks)
    pickle.dump(recs, open(path, "wb"))
    return recs


def flatten(records, classifiers=tuple(CLASSIFIERS)):
    """One row per (method, dataset, run, classifier) with the scalar metrics."""
    rows = []
    for r in records:
        for c in classifiers:
            t = r["test"][c]
            rows.append({"method": r["method"], "dataset": r["dataset"], "run": r["run"], "classifier": c,
                         "n_selected": r["n_selected"], "reduction": 1 - r["n_selected"] / r["n_total_features"],
                         "search_time_s": r.get("time_s", 0.0), "n_classifier_fits": r.get("n_classifier_fits", 0),
                         "cache_hits": r.get("cache_hits", 0), "best_fitness": r.get("best_fitness", np.nan),
                         **{k: t[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_macro")}})
    return pd.DataFrame(rows)


def load_all():
    recs = []
    for p in sorted(RAW_DIR.glob("*.pkl")):
        recs += pickle.load(open(p, "rb"))
    return recs
