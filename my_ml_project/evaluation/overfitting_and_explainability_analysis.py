"""
Phase 2 audit, Tasks 3-6: overfitting/underfitting analysis, comprehensive
metrics, feature importance / explainability (native + permutation), and
error analysis (false positives / false negatives, incl. audit-only
cross-tab against excluded demographic columns).
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_common import RANDOM_STATE, EXCLUDED_COLUMNS, load_all

warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "raw", "student_career_success_dataset.xlsx",
)
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH

df, X, y, label_encoder, numeric_features, categorical_features, preprocessor = load_all(DATA_PATH)
not_employable_idx = list(label_encoder.classes_).index("Not Employable")
employable_idx = 1 - not_employable_idx

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
# Keep index alignment to join back excluded demographic columns for audit-only error analysis
train_idx, test_idx = X_train.index, X_test.index

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4),
    "XGBoost": XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, subsample=1.0,
        colsample_bytree=1.0, min_child_weight=1,
        random_state=RANDOM_STATE, n_jobs=4, eval_metric="logloss",
    ),
}


def predict_labels(pipe, X_data):
    proba = pipe.predict_proba(X_data)
    y_pred_bin = (proba[:, not_employable_idx] >= 0.50).astype(int)
    y_pred_labels = np.where(y_pred_bin == 1, not_employable_idx, employable_idx)
    return y_pred_labels, proba[:, not_employable_idx]


def metrics_block(y_true, y_pred_labels, proba_not_emp):
    acc = accuracy_score(y_true, y_pred_labels)
    auc = roc_auc_score(y_true, proba_not_emp)
    f1_macro = f1_score(y_true, y_pred_labels, average="macro")
    f1_weighted = f1_score(y_true, y_pred_labels, average="weighted")
    bal_acc = balanced_accuracy_score(y_true, y_pred_labels)
    return {"accuracy": acc, "roc_auc": auc, "f1_macro": f1_macro,
            "f1_weighted": f1_weighted, "balanced_accuracy": bal_acc}


# ---------------------------------------------------------------------------
# Task 3 + 4: fit all 4 models, compute train vs test vs (CV mean loaded from
# task-1 json if available) metrics; comprehensive test metrics table.
# ---------------------------------------------------------------------------
cv_results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "phase2_cv_results.json")
cv_results = None
if os.path.exists(cv_results_path):
    with open(cv_results_path) as f:
        cv_results = json.load(f)

overfit_table = []
comprehensive_metrics = {}
fitted_pipelines = {}

print("=" * 70)
print("TASK 3: Overfitting / underfitting — train vs test vs CV")
print("=" * 70)

for name, clf in MODELS.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred_train, proba_train = predict_labels(pipe, X_train)
    y_pred_test, proba_test = predict_labels(pipe, X_test)

    train_m = metrics_block(y_train, y_pred_train, proba_train)
    test_m = metrics_block(y_test, y_pred_test, proba_test)

    cv_mean_acc = cv_mean_auc = None
    if cv_results:
        cv_mean_acc = cv_results["cv_full_data"][name]["accuracy"]["mean"]
        cv_mean_auc = cv_results["cv_full_data"][name]["roc_auc"]["mean"]

    gap_acc = train_m["accuracy"] - test_m["accuracy"]
    gap_auc = train_m["roc_auc"] - test_m["roc_auc"]

    verdict = "good generalization"
    if gap_acc > 0.04:
        verdict = "OVERFITTING signal (train-test acc gap > 4pp)"
    elif train_m["accuracy"] < 0.75 and test_m["accuracy"] < 0.75:
        verdict = "possible UNDERFITTING (both train & test mediocre)"

    row = {
        "model": name,
        "train_acc": train_m["accuracy"], "test_acc": test_m["accuracy"],
        "gap_acc": gap_acc, "cv_mean_acc": cv_mean_acc,
        "train_auc": train_m["roc_auc"], "test_auc": test_m["roc_auc"],
        "gap_auc": gap_auc, "cv_mean_auc": cv_mean_auc,
        "verdict": verdict,
    }
    overfit_table.append(row)
    comprehensive_metrics[name] = {"train": train_m, "test": test_m}

    print(f"\n{name}:")
    print(f"  train_acc={train_m['accuracy']:.4f}  test_acc={test_m['accuracy']:.4f}  "
          f"gap={gap_acc:+.4f}  cv_mean_acc={cv_mean_acc}")
    print(f"  train_auc={train_m['roc_auc']:.4f}  test_auc={test_m['roc_auc']:.4f}  "
          f"gap={gap_auc:+.4f}  cv_mean_auc={cv_mean_auc}")
    print(f"  verdict: {verdict}")

# ---------------------------------------------------------------------------
# Task 4: comprehensive metrics table (final XGBoost + others) on test set
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 4: Comprehensive test-set metrics")
print("=" * 70)
comprehensive_full = {}
for name, pipe in fitted_pipelines.items():
    y_pred_test, proba_test = predict_labels(pipe, X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, y_pred_test, labels=[employable_idx, not_employable_idx])
    report = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, output_dict=True)
    bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    acc = accuracy_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, proba_test)
    f1_macro = f1_score(y_test, y_pred_test, average="macro")
    f1_weighted = f1_score(y_test, y_pred_test, average="weighted")

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}  Balanced accuracy: {bal_acc:.4f}  ROC-AUC: {auc:.4f}")
    print(f"F1-macro: {f1_macro:.4f}  F1-weighted: {f1_weighted:.4f}")
    print(f"Confusion matrix [rows/cols = {list(label_encoder.classes_)}]:\n{cm}")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

    comprehensive_full[name] = {
        "accuracy": acc, "balanced_accuracy": bal_acc, "roc_auc": auc,
        "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
    }

# ---------------------------------------------------------------------------
# Task 3b: regularization experiment for XGBoost (only if meaningful gap)
# ---------------------------------------------------------------------------
xgb_row = [r for r in overfit_table if r["model"] == "XGBoost"][0]
print("\n" + "=" * 70)
print(f"XGBoost train-test acc gap = {xgb_row['gap_acc']:+.4f} "
      f"({'>3-4pp -> trying regularization' if xgb_row['gap_acc'] > 0.03 else '<=3pp -> gap is small, regularization experiment run anyway for completeness'})")
print("=" * 70)

from sklearn.model_selection import RandomizedSearchCV

reg_param_distributions = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__max_depth": [2, 3, 4],
    "classifier__subsample": [0.7, 0.8, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 1.0],
    "classifier__min_child_weight": [1, 3, 5, 7, 10],
    "classifier__reg_alpha": [0, 0.1, 0.5, 1.0],
    "classifier__reg_lambda": [1.0, 2.0, 5.0, 10.0],
}
reg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(random_state=RANDOM_STATE, n_jobs=4, eval_metric="logloss")),
])
reg_search = RandomizedSearchCV(
    reg_pipeline, param_distributions=reg_param_distributions, n_iter=30, cv=3,
    scoring="accuracy", random_state=RANDOM_STATE, n_jobs=1, verbose=1,
)
reg_search.fit(X_train, y_train)
print("Regularized search best_params_:", reg_search.best_params_)
print(f"Regularized search internal CV best score: {reg_search.best_score_:.4f}")

reg_pipe = reg_search.best_estimator_
y_pred_train_reg, proba_train_reg = predict_labels(reg_pipe, X_train)
y_pred_test_reg, proba_test_reg = predict_labels(reg_pipe, X_test)
train_m_reg = metrics_block(y_train, y_pred_train_reg, proba_train_reg)
test_m_reg = metrics_block(y_test, y_pred_test_reg, proba_test_reg)
reg_gap = train_m_reg["accuracy"] - test_m_reg["accuracy"]
print(f"Regularized XGBoost: train_acc={train_m_reg['accuracy']:.4f} test_acc={test_m_reg['accuracy']:.4f} "
      f"gap={reg_gap:+.4f} test_auc={test_m_reg['roc_auc']:.4f}")
print(f"Comparison vs original tuned XGBoost: test_acc {test_m_reg['accuracy']:.4f} vs "
      f"{comprehensive_metrics['XGBoost']['test']['accuracy']:.4f}, "
      f"test_auc {test_m_reg['roc_auc']:.4f} vs {comprehensive_metrics['XGBoost']['test']['roc_auc']:.4f}")

regularization_result = {
    "best_params": reg_search.best_params_,
    "internal_cv_best_score": float(reg_search.best_score_),
    "train_acc": float(train_m_reg["accuracy"]), "test_acc": float(test_m_reg["accuracy"]),
    "gap_acc": float(reg_gap), "test_auc": float(test_m_reg["roc_auc"]),
    "original_test_acc": float(comprehensive_metrics["XGBoost"]["test"]["accuracy"]),
    "original_test_auc": float(comprehensive_metrics["XGBoost"]["test"]["roc_auc"]),
}

# ---------------------------------------------------------------------------
# Task 5: feature importance -- native (XGBoost) + permutation importance
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 5: Feature importance -- native vs permutation")
print("=" * 70)
xgb_pipe = fitted_pipelines["XGBoost"]
fitted_preprocessor = xgb_pipe.named_steps["preprocessor"]
ohe = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_expanded_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_expanded_names)

native_importances = xgb_pipe.named_steps["classifier"].feature_importances_
native_map = {}
for name_, score in zip(all_feature_names, native_importances):
    raw_name = name_
    for cat in categorical_features:
        if name_.startswith(cat + "_"):
            raw_name = cat
            break
    native_map[raw_name] = native_map.get(raw_name, 0.0) + float(score)
native_total = sum(native_map.values()) or 1.0
native_ranked = sorted(((k, v / native_total) for k, v in native_map.items()),
                        key=lambda kv: kv[1], reverse=True)
print("\nNative XGBoost feature_importances_ (normalised, mapped to raw attributes):")
for name_, score in native_ranked:
    print(f"  {name_:30s} {score:.6f}")

print("\nRunning permutation_importance on held-out test set (n_repeats=10, scoring='roc_auc')...")
perm_result = permutation_importance(
    xgb_pipe, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
    scoring="roc_auc", n_jobs=4,
)
perm_ranked_raw = sorted(
    zip(X_test.columns, perm_result.importances_mean, perm_result.importances_std),
    key=lambda t: t[1], reverse=True,
)
print("\nPermutation importance (raw columns, scoring=roc_auc, mean +/- std over 10 repeats):")
for name_, mean_, std_ in perm_ranked_raw:
    print(f"  {name_:30s} {mean_:.6f} +/- {std_:.6f}")

native_rank_order = [n for n, _ in native_ranked]
perm_rank_order = [n for n, _, _ in perm_ranked_raw]
print("\nNative top-5:", native_rank_order[:5])
print("Permutation top-5:", perm_rank_order[:5])

# ---------------------------------------------------------------------------
# Task 6: error analysis on the final XGBoost model's test predictions
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 6: Error analysis (XGBoost, test set)")
print("=" * 70)
y_pred_xgb, proba_xgb = predict_labels(xgb_pipe, X_test)

# original label semantics: employable_idx = "Employable" index, not_employable_idx = "Not Employable"
is_fp = (y_test == employable_idx) & (y_pred_xgb == not_employable_idx)   # actually Employable, predicted Not Employable
is_fn = (y_test == not_employable_idx) & (y_pred_xgb == employable_idx)   # actually Not Employable, predicted Employable
is_tn_correct_employable = (y_test == employable_idx) & (y_pred_xgb == employable_idx)
is_tp_correct_notemp = (y_test == not_employable_idx) & (y_pred_xgb == not_employable_idx)

print(f"False positives (actually Employable, predicted Not Employable): {is_fp.sum()}")
print(f"False negatives (actually Not Employable, predicted Employable): {is_fn.sum()}")
print(f"Correctly predicted Employable: {is_tn_correct_employable.sum()}")
print(f"Correctly predicted Not Employable: {is_tp_correct_notemp.sum()}")

key_numeric = ["Resume_Score", "Interview_Score", "CGPA", "Internships", "Programming_Skill",
               "Overall_Preparedness_Index", "Skills_per_Project"]
key_numeric = [c for c in key_numeric if c in X_test.columns]

fp_slice = X_test.loc[is_fp[is_fp].index] if hasattr(is_fp, "index") else X_test[is_fp]
fn_slice = X_test.loc[is_fn[is_fn].index] if hasattr(is_fn, "index") else X_test[is_fn]
correct_employable_slice = X_test[is_tn_correct_employable]
correct_notemp_slice = X_test[is_tp_correct_notemp]

fp_slice = X_test[is_fp]
fn_slice = X_test[is_fn]

print("\n--- False positives (actually Employable) vs correctly-classified Employable ---")
error_analysis_fp = {}
for col in key_numeric:
    fp_mean, fp_med = fp_slice[col].mean(), fp_slice[col].median()
    ok_mean, ok_med = correct_employable_slice[col].mean(), correct_employable_slice[col].median()
    print(f"  {col:28s} FP: mean={fp_mean:.3f} median={fp_med:.3f}  |  Correct-Employable: mean={ok_mean:.3f} median={ok_med:.3f}")
    error_analysis_fp[col] = {"fp_mean": float(fp_mean), "fp_median": float(fp_med),
                               "correct_mean": float(ok_mean), "correct_median": float(ok_med)}

print("\n--- False negatives (actually Not Employable) vs correctly-classified Not Employable ---")
error_analysis_fn = {}
for col in key_numeric:
    fn_mean, fn_med = fn_slice[col].mean(), fn_slice[col].median()
    ok_mean, ok_med = correct_notemp_slice[col].mean(), correct_notemp_slice[col].median()
    print(f"  {col:28s} FN: mean={fn_mean:.3f} median={fn_med:.3f}  |  Correct-NotEmployable: mean={ok_mean:.3f} median={ok_med:.3f}")
    error_analysis_fn[col] = {"fn_mean": float(fn_mean), "fn_median": float(fn_med),
                               "correct_mean": float(ok_mean), "correct_median": float(ok_med)}

# Audit-only: join back excluded demographic columns for cross-tab of error rates
print("\n--- Audit-only: error rate by excluded demographic columns (NOT used in training) ---")
demo_cols_present = [c for c in EXCLUDED_COLUMNS if c in df.columns]
demo_test = df.loc[X_test.index, demo_cols_present].copy()
demo_test["y_true"] = y_test
demo_test["y_pred"] = y_pred_xgb
demo_test["is_fp"] = is_fp.values if hasattr(is_fp, "values") else is_fp
demo_test["is_fn"] = is_fn.values if hasattr(is_fn, "values") else is_fn

demographic_crosstab = {}
for demo_col in ["Major", "Gender", "University_Year"]:
    if demo_col not in demo_test.columns:
        continue
    print(f"\n  By {demo_col}:")
    rows = []
    for cat, grp in demo_test.groupby(demo_col):
        n = len(grp)
        n_actual_not_emp = (grp["y_true"] == not_employable_idx).sum()
        n_actual_emp = (grp["y_true"] == employable_idx).sum()
        fn_rate = grp["is_fn"].sum() / n_actual_not_emp if n_actual_not_emp else float("nan")
        fp_rate = grp["is_fp"].sum() / n_actual_emp if n_actual_emp else float("nan")
        overall_acc = (grp["y_true"] == grp["y_pred"]).mean()
        print(f"    {cat:25s} n={n:5d}  FN_rate(missed at-risk)={fn_rate:.4f}  "
              f"FP_rate={fp_rate:.4f}  overall_acc={overall_acc:.4f}")
        rows.append({"category": cat, "n": int(n), "fn_rate": float(fn_rate) if n_actual_not_emp else None,
                      "fp_rate": float(fp_rate) if n_actual_emp else None, "overall_acc": float(overall_acc)})
    demographic_crosstab[demo_col] = rows

# ---------------------------------------------------------------------------
# Save all results
# ---------------------------------------------------------------------------
output = {
    "overfit_table": overfit_table,
    "comprehensive_metrics_train_test": comprehensive_metrics,
    "comprehensive_full_test_metrics": comprehensive_full,
    "regularization_experiment": regularization_result,
    "native_importance_ranked": native_ranked,
    "permutation_importance_ranked": [[n, float(m), float(s)] for n, m, s in perm_ranked_raw],
    "error_analysis_false_positive": error_analysis_fp,
    "error_analysis_false_negative": error_analysis_fn,
    "counts": {
        "false_positives": int(is_fp.sum()), "false_negatives": int(is_fn.sum()),
        "correct_employable": int(is_tn_correct_employable.sum()),
        "correct_not_employable": int(is_tp_correct_notemp.sum()),
    },
    "demographic_crosstab": demographic_crosstab,
}
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "phase2_overfit_explain_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved results to {out_path}")
