# 🎓 Student Employability Prediction & Recommendation System

An end-to-end Machine Learning and Backend System that predicts student placement eligibility based on academic, technical, and interpersonal performance metrics. Built with **XGBoost**, **SMOTE**, and **Scikit-Learn** in Python, and served via a high-performance **FastAPI** REST API with real-time performance gap analysis.

---

## 📌 Features

- **Automated Preprocessing Pipeline:** Integrated imputation, scaling, and one-hot encoding using `ColumnTransformer` to eliminate data leakage.
- **Class Imbalance Handling:** Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) inside an `ImbPipeline` to balance minority class representations.
- **Advanced Classification:** Trained with an optimized **XGBoost** model utilizing custom decision thresholding ($0.35$) for maximized recall.
- **Domain Feature Engineering:** Engineered custom metrics such as `Overall_Preparedness_Index` and `Skills_per_Project`.
- **Post-Processing Gap Analysis:** Evaluates non-employable candidates against feature importance benchmarks to generate real-time feedback.
- **Production-Ready REST API:** Clean and validated endpoints built with **FastAPI**.

---

## 🏗️ System Architecture & Lifecycle

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    1. MODEL BUILDING (Google Colab)                     │
│ Data Ingestion ➔ Feature Engineering ➔ Preprocessing ➔ SMOTE ➔ XGBoost │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Export Artifacts (.pkl)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     2. API INFERENCE (FastAPI)                          │
│ Client Request ➔ Model Load ➔ Feature Transform ➔ Predict Probabilities │
│ ➔ Thresholding (0.35) ➔ Gap Analysis Engine ➔ JSON Response             │
└─────────────────────────────────────────────────────────────────────────┘



