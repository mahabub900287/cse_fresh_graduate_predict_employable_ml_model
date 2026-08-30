"""
Phase 2 audit — shared data loading helper.

Copies EXACTLY the data loading / cleaning / feature-engineering / column
exclusion logic from train_model.py (do not modify train_model.py itself;
this module is read-only support code for the new Phase 2 analysis scripts).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_COL = "Placement_Status"
EXCLUDED_COLUMNS = [
    "Age",
    "Gender",
    "University_Year",
    "Major",
    "Attendance_Percentage",
    "LinkedIn_Profile",
]
LEAKAGE_COLS = [
    "Student_ID", "Employability_Score", "Company_Tier", "Career_Field",
    "Placement_Mode", "Starting_Salary_USD",
]


def load_clean_dataframe(data_path):
    """Reproduces train_model.py lines 51-90 exactly."""
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    if set(df[TARGET_COL].unique()) <= {"Placed", "Not Placed"}:
        df[TARGET_COL] = df[TARGET_COL].map(
            {"Placed": "Employable", "Not Placed": "Not Employable"}
        )

    assert TARGET_COL in df.columns, f"Target column {TARGET_COL} not found in dataset"

    leakage_cols = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=leakage_cols)

    df["Overall_Preparedness_Index"] = (
        df["Interview_Score"] * 0.7 + df["Internships"] * 0.3
    )
    df["Skills_per_Project"] = df["Programming_Skill"] / (df["Projects_Completed"] + 1)

    return df


def build_xy(df):
    """Returns X (18 features), y (encoded), label_encoder, numeric_features,
    categorical_features -- mirrors train_model.py lines 95-105."""
    y_raw = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + EXCLUDED_COLUMNS)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, label_encoder, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
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
    return preprocessor


def load_all(data_path):
    df = load_clean_dataframe(data_path)
    X, y, label_encoder, numeric_features, categorical_features = build_xy(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return df, X, y, label_encoder, numeric_features, categorical_features, preprocessor
