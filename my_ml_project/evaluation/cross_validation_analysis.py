"""
Phase 2 audit, Task 1: Stratified 5-fold cross-validation robustness check.

Runs StratifiedKFold(n_splits=5, shuffle=True, random_state=42) for each of
the 4 compared algorithms (+ LightGBM/CatBoost if installed), both on the
FULL labeled dataset (X, y) and on the train-only partition (X_train,
y_train) from the identical 80:20 split used in train_model.py.

Metrics per fold: accuracy, ROC-AUC, F1-macro, F1-weighted, minority-class
(Not Employable) recall & precision. Reports mean +/- std per model.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_common import RANDOM_STATE, load_all

warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "raw", "student_career_success_dataset.xlsx",
)
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH

HAS_LGBM = False
HAS_CATBOOST = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    pass
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    pass

print(f"LightGBM available: {HAS_LGBM}; CatBoost available: {HAS_CATBOOST}")

df, X, y, label_encoder, numeric_features, categorical_features, preprocessor = load_all(DATA_PATH)
not_employable_idx = list(label_encoder.classes_).index("Not Employable")
print("Label classes:", list(label_encoder.classes_), "-> Not Employable idx:", not_employable_idx)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)


def make_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, subsample=1.0,
            colsample_bytree=1.0, min_child_weight=1,
            random_state=RANDOM_STATE, n_jobs=4, eval_metric="logloss",
        ),
    }
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=4, verbose=-1)
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(random_state=RANDOM_STATE, verbose=False)
    return models


def build_preproc_clone():
    from phase2_common import build_preprocessor
    return build_preprocessor(numeric_features, categorical_features)


def run_cv(X_data, y_data, label):
    print(f"\n{'='*70}\nCV on: {label}  (n={X_data.shape[0]})\n{'='*70}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_results = {}
    for name, clf in make_models().items():
        pipe = Pipeline([("preprocessor", build_preproc_clone()), ("classifier", clf)])
        fold_metrics = {"accuracy": [], "roc_auc": [], "f1_macro": [], "f1_weighted": [],
                         "minority_recall": [], "minority_precision": []}
        t0 = time.time()
        for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_data, y_data)):
            X_tr = X_data.iloc[tr_idx]
            X_te = X_data.iloc[te_idx]
            y_tr = y_data[tr_idx]
            y_te = y_data[te_idx]

            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)
            y_pred_bin = (proba[:, not_employable_idx] >= 0.50).astype(int)
            other_idx = 1 - not_employable_idx
            y_pred_labels = np.where(y_pred_bin == 1, not_employable_idx, other_idx)

            fold_metrics["accuracy"].append(accuracy_score(y_te, y_pred_labels))
            fold_metrics["roc_auc"].append(roc_auc_score(y_te, proba[:, not_employable_idx]))
            fold_metrics["f1_macro"].append(f1_score(y_te, y_pred_labels, average="macro"))
            fold_metrics["f1_weighted"].append(f1_score(y_te, y_pred_labels, average="weighted"))
            fold_metrics["minority_recall"].append(
                recall_score(y_te, y_pred_labels, pos_label=not_employable_idx))
            fold_metrics["minority_precision"].append(
                precision_score(y_te, y_pred_labels, pos_label=not_employable_idx, zero_division=0))
        elapsed = time.time() - t0

        summary = {}
        for metric, values in fold_metrics.items():
            arr = np.array(values)
            summary[metric] = {"mean": float(arr.mean()), "std": float(arr.std()), "folds": values}
        all_results[name] = summary
        print(f"\n{name} (fit+eval 5 folds took {elapsed:.1f}s):")
        for metric, s in summary.items():
            print(f"  {metric:20s} mean={s['mean']:.4f}  std={s['std']:.4f}  folds={[round(v,4) for v in s['folds']]}")
    return all_results


results_full = run_cv(X, y, "FULL DATASET (X, y)")
results_train_only = run_cv(X_train, y_train, "TRAIN-ONLY PARTITION (X_train, y_train)")

output = {
    "not_employable_idx": int(not_employable_idx),
    "label_classes": list(label_encoder.classes_),
    "cv_full_data": results_full,
    "cv_train_only": results_train_only,
}
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "phase2_cv_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved results to {out_path}")

# ---------------------------------------------------------------------------
# Compare single-split test performance (compare_algorithms.py) against
# CV mean +/- 2*std to flag lucky/unlucky splits.
# ---------------------------------------------------------------------------
SINGLE_SPLIT_TEST = {
    "Logistic Regression": {"accuracy": 0.8196, "roc_auc": 0.8094},
    "Decision Tree": {"accuracy": 0.8109, "roc_auc": 0.7917},
    "Random Forest": {"accuracy": 0.8138, "roc_auc": 0.7984},
    "XGBoost": {"accuracy": 0.8170, "roc_auc": 0.8068},
}

print(f"\n{'='*70}\nSingle-split-vs-CV flag check (full-data CV)\n{'='*70}")
for name, single in SINGLE_SPLIT_TEST.items():
    cv = results_full[name]
    for metric in ("accuracy", "roc_auc"):
        mean = cv[metric]["mean"]
        std = cv[metric]["std"]
        val = single[metric]
        lo, hi = mean - 2 * std, mean + 2 * std
        flag = "OUTSIDE 2*std" if not (lo <= val <= hi) else "within range"
        rel_std = std / mean if mean else float("nan")
        instability_flag = "UNSTABLE (std>1% of mean)" if rel_std > 0.01 else "stable"
        print(f"{name:22s} {metric:10s} single={val:.4f} cv_mean={mean:.4f} cv_std={std:.4f} "
              f"[{lo:.4f},{hi:.4f}] -> {flag}; rel_std={rel_std:.4f} -> {instability_flag}")
