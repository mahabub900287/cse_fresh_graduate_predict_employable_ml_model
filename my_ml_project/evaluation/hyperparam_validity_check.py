"""
Phase 2 audit, Task 2: Hyperparameter search validity re-check.

Part A: Rerun train_model.py's exact RandomizedSearchCV (n_iter=15, cv=3,
        scoring='accuracy') fresh and confirm best_params_ reproduces
        (random_state=42 -> should be deterministic).
Part B: Run a WIDER RandomizedSearchCV (n_iter=40, same grid, same cv=3) on
        X_train only, then evaluate the chosen model ONCE on X_test (never
        used inside the search) and compare to the original n_iter=15 result.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_common import RANDOM_STATE, load_all

warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "raw", "student_career_success_dataset.xlsx",
)
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH

df, X, y, label_encoder, numeric_features, categorical_features, preprocessor = load_all(DATA_PATH)
not_employable_idx = list(label_encoder.classes_).index("Not Employable")
other_idx = 1 - not_employable_idx

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

ORIGINAL_BEST_PARAMS = {
    "classifier__n_estimators": 200,
    "classifier__learning_rate": 0.05,
    "classifier__max_depth": 3,
    "classifier__subsample": 1.0,
    "classifier__colsample_bytree": 1.0,
    "classifier__min_child_weight": 1,
}
ORIGINAL_TEST_ACC = 0.8170
ORIGINAL_TEST_AUC = 0.8068


def eval_on_test(fitted_pipeline, label):
    proba = fitted_pipeline.predict_proba(X_test)
    y_pred_bin = (proba[:, not_employable_idx] >= 0.50).astype(int)
    y_pred_labels = np.where(y_pred_bin == 1, not_employable_idx, other_idx)
    acc = accuracy_score(y_test, y_pred_labels)
    auc = roc_auc_score(y_test, proba[:, not_employable_idx])
    cm = confusion_matrix(y_test, y_pred_labels)
    print(f"\n--- {label}: test evaluation ---")
    print(f"Accuracy: {acc:.4f}  ROC-AUC: {auc:.4f}")
    print("Confusion matrix:\n", cm)
    print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_))
    return acc, auc, cm


def make_pipeline():
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(random_state=RANDOM_STATE, n_jobs=4, eval_metric="logloss")),
    ])


param_distributions = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__max_depth": [3, 4, 6],
    "classifier__subsample": [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0],
    "classifier__min_child_weight": [1, 3, 5],
}

# ---------------------------------------------------------------------------
# Part A: reproduce original n_iter=15 search exactly
# ---------------------------------------------------------------------------
print("=" * 70)
print("PART A: Reproducing original RandomizedSearchCV (n_iter=15, cv=3)")
print("=" * 70)
pipeline_a = make_pipeline()
search_a = RandomizedSearchCV(
    pipeline_a, param_distributions=param_distributions, n_iter=15, cv=3,
    scoring="accuracy", random_state=RANDOM_STATE, n_jobs=1, verbose=1,
)
t0 = time.time()
search_a.fit(X_train, y_train)
print(f"Search A completed in {time.time()-t0:.1f}s")
print("Reproduced best_params_:", search_a.best_params_)
print("Original stated best_params_:", ORIGINAL_BEST_PARAMS)
match = search_a.best_params_ == ORIGINAL_BEST_PARAMS
print(f"EXACT MATCH: {match}")

acc_a, auc_a, cm_a = eval_on_test(search_a.best_estimator_, "Part A reproduced n_iter=15 model")

# ---------------------------------------------------------------------------
# Part B: wider search, n_iter=40, same grid, cv=3, X_train only
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART B: Wider RandomizedSearchCV (n_iter=40, cv=3) on X_train only")
print("=" * 70)
pipeline_b = make_pipeline()
search_b = RandomizedSearchCV(
    pipeline_b, param_distributions=param_distributions, n_iter=40, cv=3,
    scoring="accuracy", random_state=RANDOM_STATE, n_jobs=1, verbose=1,
)
t0 = time.time()
search_b.fit(X_train, y_train)
print(f"Search B completed in {time.time()-t0:.1f}s")
print("Wider-search best_params_:", search_b.best_params_)
print(f"Wider-search best CV accuracy (internal cv=3): {search_b.best_score_:.4f}")
print(f"Original n_iter=15 best CV accuracy (internal cv=3): {search_a.best_score_:.4f}")

acc_b, auc_b, cm_b = eval_on_test(search_b.best_estimator_, "Part B wider n_iter=40 model")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Original (thesis-stated) test accuracy/AUC: {ORIGINAL_TEST_ACC:.4f} / {ORIGINAL_TEST_AUC:.4f}")
print(f"Part A reproduced (n_iter=15)  test accuracy/AUC: {acc_a:.4f} / {auc_a:.4f}  params_match={match}")
print(f"Part B wider search (n_iter=40) test accuracy/AUC: {acc_b:.4f} / {auc_b:.4f}")
print(f"Delta (wider - original n_iter=15), accuracy: {acc_b-acc_a:+.4f}   AUC: {auc_b-auc_a:+.4f}")

output = {
    "part_a_reproduced_best_params": search_a.best_params_,
    "part_a_matches_original": bool(match),
    "part_a_internal_cv_best_score": float(search_a.best_score_),
    "part_a_test_acc": float(acc_a),
    "part_a_test_auc": float(auc_a),
    "part_b_wider_best_params": search_b.best_params_,
    "part_b_internal_cv_best_score": float(search_b.best_score_),
    "part_b_test_acc": float(acc_b),
    "part_b_test_auc": float(auc_b),
    "original_stated_best_params": ORIGINAL_BEST_PARAMS,
    "original_stated_test_acc": ORIGINAL_TEST_ACC,
    "original_stated_test_auc": ORIGINAL_TEST_AUC,
}
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "phase2_hyperparam_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved results to {out_path}")
