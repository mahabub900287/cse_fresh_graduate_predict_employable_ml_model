"""
Table 5.6: Comparison of candidate classification algorithms for this problem.

Trains Logistic Regression, Decision Tree, Random Forest and XGBoost on the
identical stratified 80:20 split, the same 18-feature predictor matrix, and
the same default 0.50 decision threshold, as described in Section 5.8 of the
report. Support Vector Machine is excluded, as in the report, on
computational-cost grounds at n = 50,000.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "student_career_success_dataset.csv"

EXCLUDED_COLUMNS = [
    "Age", "Gender", "University_Year", "Major",
    "Attendance_Percentage", "LinkedIn_Profile",
]
TARGET_COL = "Placement_Status"

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

if set(df[TARGET_COL].unique()) <= {"Placed", "Not Placed"}:
    df[TARGET_COL] = df[TARGET_COL].map({"Placed": "Employable", "Not Placed": "Not Employable"})

leakage_cols = [c for c in ["Student_ID", "Employability_Score", "Company_Tier",
                             "Career_Field", "Placement_Mode", "Starting_Salary_USD"]
                if c in df.columns]
df = df.drop(columns=leakage_cols)

df["Overall_Preparedness_Index"] = df["Interview_Score"] * 0.7 + df["Internships"] * 0.3
df["Skills_per_Project"] = df["Programming_Skill"] / (df["Projects_Completed"] + 1)

y_raw = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL] + EXCLUDED_COLUMNS)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
not_employable_idx = list(label_encoder.classes_).index("Not Employable")

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

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

print(f"{'Model':<22}{'Accuracy':>10}{'ROC-AUC':>10}{'Train (s)':>12}")
print("-" * 54)

results = []
for name, clf in MODELS.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    elapsed = time.time() - t0

    proba = pipe.predict_proba(X_test)
    y_pred = (proba[:, not_employable_idx] >= 0.50).astype(int)
    other_idx = 1 - not_employable_idx
    y_pred_labels = np.where(y_pred == 1, not_employable_idx, other_idx)

    acc = accuracy_score(y_test, y_pred_labels)
    auc = roc_auc_score(y_test, proba[:, not_employable_idx])
    results.append((name, acc, auc, elapsed))
    print(f"{name:<22}{acc:>10.4f}{auc:>10.4f}{elapsed:>12.2f}")

print("\n=== Markdown table ===")
print("| Algorithm | Accuracy | ROC-AUC | Training time (s) |")
print("|---|---|---|---|")
for name, acc, auc, elapsed in results:
    print(f"| {name} | {acc:.4f} | {auc:.4f} | {elapsed:.2f} |")
