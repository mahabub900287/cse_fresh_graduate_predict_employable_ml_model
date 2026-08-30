"""
Phase 2 audit, Task 7: multi-seed stability check.

Re-splits the data 80:20 at several additional random seeds (7, 123, 2024),
refits the FINAL XGBoost pipeline with the SAME tuned hyperparameters
(no re-tuning), and reports test accuracy/AUC spread vs. the original
random_state=42 split.
"""
import json
import os
import sys
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_common import load_all

warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "raw", "student_career_success_dataset.xlsx",
)
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH

df, X, y, label_encoder, numeric_features, categorical_features, preprocessor = load_all(DATA_PATH)
not_employable_idx = list(label_encoder.classes_).index("Not Employable")
employable_idx = 1 - not_employable_idx

SEEDS = [42, 7, 123, 2024]  # 42 = original, others additional

BEST_PARAMS = dict(
    n_estimators=200, learning_rate=0.05, max_depth=3, subsample=1.0,
    colsample_bytree=1.0, min_child_weight=1,
)

results = []
for seed in SEEDS:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            **BEST_PARAMS, random_state=42, n_jobs=4, eval_metric="logloss",
        )),
    ])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)
    y_pred_bin = (proba[:, not_employable_idx] >= 0.50).astype(int)
    y_pred_labels = np.where(y_pred_bin == 1, not_employable_idx, employable_idx)

    acc = accuracy_score(y_test, y_pred_labels)
    auc = roc_auc_score(y_test, proba[:, not_employable_idx])
    print(f"split_seed={seed:5d}  test_acc={acc:.4f}  test_auc={auc:.4f}")
    results.append({"split_seed": seed, "test_acc": float(acc), "test_auc": float(auc)})

accs = np.array([r["test_acc"] for r in results])
aucs = np.array([r["test_auc"] for r in results])
print(f"\nAcross {len(SEEDS)} seeds: acc mean={accs.mean():.4f} std={accs.std():.4f} "
      f"min={accs.min():.4f} max={accs.max():.4f} range={accs.max()-accs.min():.4f}")
print(f"Across {len(SEEDS)} seeds: auc mean={aucs.mean():.4f} std={aucs.std():.4f} "
      f"min={aucs.min():.4f} max={aucs.max():.4f} range={aucs.max()-aucs.min():.4f}")

output = {
    "per_seed": results,
    "summary": {
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
        "acc_min": float(accs.min()), "acc_max": float(accs.max()),
        "auc_mean": float(aucs.mean()), "auc_std": float(aucs.std()),
        "auc_min": float(aucs.min()), "auc_max": float(aucs.max()),
    },
}
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "phase2_multiseed_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved results to {out_path}")
