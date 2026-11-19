# binarymetaheuristic-featureselection

Experiments for a **Unified Binarization Framework (UBF)** that turns continuous metaheuristics into wrapper feature
selectors. 47 continuous algorithms of [`metaheuristicpy`](https://pypi.org/project/metaheuristicpy/) 3.2.1 are
binarised by the same framework and compared with a random-search control and three classical baselines.

Author: I Made Satria Bimantara.

## Framework

| Component | What it does |
|---|---|
| Sign mapping | feature *j* is selected when `x_j > 0`; a position always gives the same subset |
| Density-diverse initial population | agent *i* draws its own target share of selected features, `p_i ~ U(0.02, 0.98)`, `x = logit(p_i) + N(0, 1)` |
| Fitness memoisation | a subset that was already scored is not trained again |
| Elite bit-flip local search | 10% of the evaluation budget on the final best subset |
| Random-search control (RS) | the same budget spent on independent draws from the same sampler |

## Protocol

- 8 UCI datasets (`datasets/`): wine, wdbc, ionosphere, sonar, vehicle, dermatology, spectf, musk1 (identifier columns
  of musk1 removed). Stratified 70/30 split (seed 2025); min-max scaling fitted on the training split.
- The wrapper search sees only the training split: fitness `0.99·(1 − F1_macro) + 0.01·|S|/D`, F1 from 3-fold CV of
  KNN (k = 5); 20 agents × 50 iterations = 1,000 evaluations; 10 independent runs.
- The selected subset is scored once on the test split with KNN, decision tree, SVM, MLP, random forest and XGBoost:
  accuracy, macro precision/recall/F1, ROC-AUC, confusion matrix, search time.
- Baselines: all features, `SelectKBest` with ANOVA-F and with chi² (k tuned by the same CV on train), and RS.
- Statistics: Friedman test over (dataset, classifier) blocks; Wilcoxon signed-rank with Holm correction.

## Results (means over 8 datasets × 6 classifiers × 10 runs)

| Method | F1-macro | Accuracy | ROC-AUC | Selected features | Search time (s) | Mean rank |
|---|---:|---:|---:|---:|---:|---:|
| All features | 0.8394 | 0.8701 | 0.9169 | 49.8 | 0.0 | 18.31 |
| BINFO | 0.8365 | 0.8652 | 0.9097 | 22.0 | 8.7 | 19.82 |
| BHOA | 0.8366 | 0.8675 | 0.9084 | 30.7 | 9.1 | 20.07 |
| BFATA | 0.8352 | 0.8655 | 0.9057 | 21.5 | 11.5 | 20.08 |
| Chi² | 0.8263 | 0.8579 | 0.9008 | 35.0 | 0.3 | 27.28 |
| ANOVA-F | 0.8272 | 0.8591 | 0.9024 | 24.6 | 0.3 | 28.51 |
| Random search (RS) | 0.8297 | 0.8603 | 0.9048 | 18.1 | 16.1 | 31.42 |

All 51 methods: `results/tables/overall_ranking.csv`. Friedman χ² = 174.9, p = 9.3e-16.

- **Search vs random search.** 38 of the 47 algorithms reach a significantly better fitness than RS before the local
  search (80 paired runs, Wilcoxon + Holm); ECO, AHO, GCRA, MO, HOA, HHO, WOA, EPO and EEGWO do not
  (`search_vs_random.csv`).
- **Test split.** INFO, HOA and FATA are significantly better than RS on test F1 (Holm p = 0.007, 0.036, 0.023) and
  are not significantly different from each other. Their advantage over the ANOVA-F and chi² filters and their gap to
  all features are **not** significant after Holm correction (`wilcoxon_top_group.csv`).
- **Ablation** (INFO and HOA, `ablation.csv`). A stochastic transfer function instead of the sign mapping lowers test
  F1 by 0.35–0.54 points (uncorrected p = 0.042 only for HOA with the time-varying sigmoid). Memoisation leaves the
  result unchanged and makes the search 2.5–2.9× faster. The diverse initialisation and the local search have no
  significant effect on test F1.

`results_v1_stochastic_ubf/` keeps the tables of the first run, which used a stochastic time-varying sigmoid: there
the search phase was no better than random search (`ubf_diagnosis.csv`), which led to the framework above.

## Layout

```text
datasets/     UCI datasets as CSV
fsbench/      experiment.py (protocol, cache), plots.py
scripts/      make_notebooks.py, run_notebooks.py, comparison_cells.py, ablation_cells.py
notebooks/    00 protocol · 01 baselines · 02–49 one per algorithm · 90 comparison · 91 ablation (executed)
results/      raw/ (cached runs), tables/ (CSV), figures/
```

## Reproduce

```bash
python -m pip install -r requirements.txt          # Python 3.11
python scripts/make_notebooks.py
FSBENCH_JOBS=3 python scripts/run_notebooks.py     # cached runs in results/raw are reused; delete a file to recompute
```

Datasets: UCI Machine Learning Repository (https://archive.ics.uci.edu).
