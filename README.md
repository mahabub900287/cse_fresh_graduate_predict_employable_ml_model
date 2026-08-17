# 🎓 Student Employability Prediction & Recommendation System

An end-to-end Machine Learning and Backend System that predicts student placement eligibility based on academic, technical, and interpersonal performance metrics. Built with **XGBoost**, **SMOTE**, and **Scikit-Learn** in Python, and served via a high-performance **FastAPI** REST API with real-time performance gap analysis.

---

## 📌 Features

- **Automated Preprocessing Pipeline:** Integrated imputation, scaling, and one-hot encoding using `ColumnTransformer` to eliminate data leakage.
- **Class Imbalance Handling:** Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) inside an `ImbPipeline` to balance minority class representations.
- **Advanced Classification:** Trained with an optimized **XGBoost** model utilizing custom decision thresholding (0.35) for maximized recall.
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
```

---

## 📜 Full Process & ML Methodology

### 1. Data Cleaning & Preprocessing

- **Duplicate Removal:** `df.drop_duplicates()` applied to ensure record uniqueness.
- **String Trimming:** `str.strip()` applied across all string columns to resolve formatting mismatches.
- **Preprocessing Pipeline:** Combined via `ColumnTransformer`:
  - **Numerical Pipeline:** `SimpleImputer(strategy='median')` ➔ `StandardScaler()`
  - **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` ➔ `OneHotEncoder(handle_unknown='ignore')`

### 2. Custom Feature Engineering

- **Overall_Preparedness_Index:** Synthesizes overall student readiness by combining weighted interview and practical metrics:

  ```
  Overall_Preparedness_Index = (Interview_Score × 0.7) + (Internships × 0.3)
  ```

- **Skills_per_Project:** Evaluates programming efficiency per completed practical project:

  ```
  Skills_per_Project = Programming_Skill / (Projects_Completed + 1)
  ```

### 3. Model Training & Imbalance Strategy

- **Split Strategy:** Stratified 80/20 train-test split (`stratify=y`).
- **Pipeline Integration:** `imbalanced-learn`'s `ImbPipeline` wraps preprocessing, SMOTE oversampling, and classification to avoid leakage during cross-validation.
- **XGBoost Classifier:** Hyperparameters configured with `n_estimators=200`, `learning_rate=0.05`, `max_depth=6`, and tuned `scale_pos_weight`.

### 4. Custom Decision Boundary & Feature Importance

- **Threshold Selection:** Evaluated probability distributions to set an optimal threshold at **0.35** for maximized F1-score and Recall.
- **Feature Importance:** Evaluated using `permutation_importance` to identify top predictive drivers powering the gap analysis recommendations.

---

## 📁 Repository & Project Structure

```
.
├── best_employability_pipeline.pkl      # Serialized Scikit-Learn + SMOTE + XGBoost pipeline
├── label_encoder.pkl                    # Serialized LabelEncoder artifact
├── main.py                              # FastAPI backend application & endpoint logic
├── student_career_success_dataset.csv   # Dataset used for model training
├── requirements.txt                     # Project dependencies
└── README.md                            # Complete Project Documentation
```

---

## 📊 Dataset Features & Schema

| Feature Name | Type | Description |
|---|---|---|
| `Age` | Integer | Student age in years |
| `Gender` | Categorical | Male / Female |
| `CGPA` | Float | Cumulative Grade Point Average (0.00 - 4.00) |
| `Internships` | Integer | Total completed internships |
| `Projects_Completed` | Integer | Total software/engineering projects built |
| `Programming_Skill` | Float | Self-assessed skill score (1.0 - 10.0) |
| `Interview_Score` | Float | Mock/Real interview evaluation score (0.0 - 100.0) |
| `Communication_Skills` | Float | Soft skill rating (1.0 - 10.0) |
| `Certifications` | Integer | Professional certifications completed |
| `GitHub_Profile` | Categorical | Yes / No |
| `Employability_Status` | Categorical (Target) | Employable / Not Employable |

---

## 🛠️ Tech Stack & Dependencies

- **Programming Language:** Python 3.10+
- **Machine Learning & Analytics:** `scikit-learn`, `xgboost`, `imbalanced-learn`, `pandas`, `numpy`, `joblib`
- **API Framework:** `FastAPI`, `Uvicorn`, `pydantic`
- **Visualization (Notebook):** `matplotlib`, `seaborn`

---

## 🚀 Setup & Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/student-employability-system.git
cd student-employability-system
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Model Artifacts

Ensure that `best_employability_pipeline.pkl` and `label_encoder.pkl` are located in the root directory alongside `main.py`.

### 5. Run the FastAPI Server

```bash
uvicorn main:app --reload
```

The server will start locally at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## 🧪 API Endpoints & Testing

Access the interactive Swagger UI at:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Endpoint: `POST /predict`

**Sample Input Payload (JSON):**

```json
{
  "Age": 22,
  "Gender": "Male",
  "CGPA": 3.2,
  "Internships": 1,
  "Projects_Completed": 3,
  "Programming_Skill": 6.5,
  "Interview_Score": 75.0,
  "Communication_Skills": 7.0,
  "Certifications": 1,
  "GitHub_Profile": "Yes"
}
```

**Sample Response 1: Not Employable (With Gap Analysis)**

```json
{
  "status": "Success",
  "prediction": "Not Employable",
  "employability_probability": 0.2845,
  "gaps_identified": [
    "Low Interview Score (75.0/100). Target at least 85+ by practicing mock interviews.",
    "Insufficient Internships (1 completed). Aim for at least 2 internships to boost practical experience.",
    "Low Project Count (3 completed). Build at least 5+ real-world projects.",
    "Programming Skill Rating (6.5/10) needs improvement. Focus on Core DSA and Problem Solving.",
    "Low Certifications count (1). Complete 3+ industry-recognized certifications.",
    "Current CGPA is 3.2. Try to maintain a CGPA above 3.5."
  ],
  "message": "Here are the areas you need to improve to become employable."
}
```

**Sample Response 2: Employable**

```json
{
  "status": "Success",
  "prediction": "Employable",
  "employability_probability": 0.8120,
  "gaps_identified": [],
  "message": "Congratulations! You meet the employability requirements."
}
```

---

## 📈 Model Evaluation Summary

- **Primary Classification Threshold:** 0.35
- **Core Evaluation Metrics:** High Recall and F1-score achieved on minority class (Employable).
- **Key Determinants:** Interview Score, Overall Preparedness Index, CGPA, and Programming Skill.

---

## 📤 GitHub Deployment Quick Commands

To push this complete project to GitHub, run the following commands in your terminal:

```bash
git init
git add .
git commit -m "feat: complete machine learning pipeline and FastAPI backend integration"
git branch -M main
git remote add origin https://github.com/your-username/student-employability-system.git
git push -u origin main
```

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.