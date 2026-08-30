"""
Trains the employability prediction pipeline exactly as described in the
project report (Report-02-UPDATED.docx), Chapter 4 (Machine Learning Model
Workflow) and Section 5.6/5.7 (tuned XGBoost configuration: no SMOTE, no
scale_pos_weight, default 0.50 decision threshold, hyperparameters selected
by RandomizedSearchCV).

Input:  dataset/raw/student_career_success_dataset.csv (the raw Kaggle
        dataset; this script cleans and normalises it in-memory) or a
        path passed as the first command-line argument.
Output: models/best_employability_pipeline.pkl, models/label_encoder.pkl
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "raw", "student_career_success_dataset.xlsx"
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH

# Attributes excluded from training (Section 5.2): demographic/contextual
# descriptors a student has no control over, plus optional attributes not
# consistently interpretable across institutions.
EXCLUDED_COLUMNS = [
    "Age",
    "Gender",
    "University_Year",
    "Major",
    "Attendance_Percentage",
    "LinkedIn_Profile",
]

TARGET_COL = "Placement_Status"

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_excel(DATA_PATH) if str(DATA_PATH).lower().endswith((".xlsx", ".xls")) else pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------------------
# Section 5.2/5.3: cleaning
# ---------------------------------------------------------------------------
df = df.drop_duplicates()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# Normalise the raw Kaggle target ("Placed"/"Not Placed") to the report's
# "Employable"/"Not Employable" labels if needed.
if set(df[TARGET_COL].unique()) <= {"Placed", "Not Placed"}:
    df[TARGET_COL] = df[TARGET_COL].map(
        {"Placed": "Employable", "Not Placed": "Not Employable"}
    )

assert TARGET_COL in df.columns, f"Target column {TARGET_COL} not found in dataset"

# Drop columns that only exist downstream of the placement outcome itself
# (post-outcome leakage) and are not part of the 23-column dataset described
# in Table 5.2 of the report.
leakage_cols = [
    c
    for c in ["Student_ID", "Employability_Score", "Company_Tier", "Career_Field",
              "Placement_Mode", "Starting_Salary_USD"]
    if c in df.columns
]
df = df.drop(columns=leakage_cols)

print(f"Loaded {df.shape[0]} instances and {df.shape[1]} columns "
      f"(after dropping {len(leakage_cols)} post-outcome columns).")

# ---------------------------------------------------------------------------
# Section 5.2: Feature Engineering — two composite features
# ---------------------------------------------------------------------------
df["Overall_Preparedness_Index"] = (
    df["Interview_Score"] * 0.7 + df["Internships"] * 0.3
)
df["Skills_per_Project"] = df["Programming_Skill"] / (df["Projects_Completed"] + 1)

# ---------------------------------------------------------------------------
# Split X / y (Section 5.2/5.6): drop target + excluded columns -> 18 features
# ---------------------------------------------------------------------------
y_raw = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL] + EXCLUDED_COLUMNS)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
print("Label classes (alphabetical):", list(label_encoder.classes_))

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
print(f"Retained {X.shape[1]} features -> {len(numeric_features)} numeric, "
      f"{len(categorical_features)} categorical.")

# ---------------------------------------------------------------------------
# Section 5.5: stratified 80:20 split, seeded
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

# ---------------------------------------------------------------------------
# Section 5.3: preprocessing pipeline (fit inside the pipeline -> no leakage)
# ---------------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ---------------------------------------------------------------------------
# Section 5.6: tuned XGBoost pipeline — no SMOTE, no scale_pos_weight
# ---------------------------------------------------------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=4,
        eval_metric="logloss",
    )),
])

param_distributions = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__max_depth": [3, 4, 6],
    "classifier__subsample": [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0],
    "classifier__min_child_weight": [1, 3, 5],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=15,
    cv=3,
    scoring="accuracy",
    random_state=RANDOM_STATE,
    n_jobs=1,
    verbose=1,
)

print("\nRunning RandomizedSearchCV (15 candidates, 3-fold CV)...")
search_start = time.time()
search.fit(X_train, y_train)
search_elapsed = time.time() - search_start
print(f"Search completed in {search_elapsed:.1f} seconds.")
print("Best params:", search.best_params_)

best_pipeline = search.best_estimator_

# Re-fit timing on the full training partition with the winning configuration
fit_start = time.time()
best_pipeline.fit(X_train, y_train)
fit_elapsed = time.time() - fit_start
print(f"Final fit on {X_train.shape[0]} training records took {fit_elapsed:.2f} seconds.")

# ---------------------------------------------------------------------------
# Section 5.7: evaluation at the default 0.50 threshold
# ---------------------------------------------------------------------------
not_employable_idx = list(label_encoder.classes_).index("Not Employable")
proba_test = best_pipeline.predict_proba(X_test)
y_pred = (proba_test[:, not_employable_idx] >= 0.50).astype(int)
# map back: predicted class index equals not_employable_idx when flagged, else the other class
other_idx = 1 - not_employable_idx
y_pred_labels = np.where(y_pred == 1, not_employable_idx, other_idx)

acc = accuracy_score(y_test, y_pred_labels)
auc = roc_auc_score(y_test, proba_test[:, not_employable_idx])
cm = confusion_matrix(y_test, y_pred_labels)

print("\n=== Classification report (threshold = 0.50, no SMOTE) ===")
print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_))
print("Confusion matrix:\n", cm)
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC (Not Employable probability): {auc:.4f}")

# ---------------------------------------------------------------------------
# Section 5.10: native feature importance, mapped back to raw attributes
# ---------------------------------------------------------------------------
fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
ohe = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_expanded_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_expanded_names)

importances = best_pipeline.named_steps["classifier"].feature_importances_
importance_map = {}
for name, score in zip(all_feature_names, importances):
    raw_name = name
    for cat in categorical_features:
        if name.startswith(cat + "_"):
            raw_name = cat
            break
    importance_map[raw_name] = importance_map.get(raw_name, 0.0) + float(score)

total = sum(importance_map.values()) or 1.0
ranked = sorted(
    ((k, v / total) for k, v in importance_map.items()),
    key=lambda kv: kv[1],
    reverse=True,
)
print("\n=== Feature importance (normalised, mapped to raw attributes) ===")
for name, score in ranked:
    print(f"{name:30s} {score:.6f}")

# ---------------------------------------------------------------------------
# Persist artefacts (Section 5.9)
# ---------------------------------------------------------------------------
os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, "best_employability_pipeline.pkl")
encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
joblib.dump(best_pipeline, model_path)
joblib.dump(label_encoder, encoder_path)
print(f"\nSaved {model_path} and {encoder_path}")
