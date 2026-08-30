# 🎓 Fresh CSE Graduate Employability Prediction System

An end-to-end Machine Learning and backend system that predicts the employability of fresh Computer Science & Engineering (CSE) graduates based on academic performance, technical skills, project/internship experience, and soft skills. Built with a tuned **XGBoost** classifier inside a leakage-safe **Scikit-Learn** pipeline, served via a **FastAPI** REST API, with a local browser-based web view for interactive testing.

All project code, data, models, evaluation results, and the thesis report/book live under [my_ml_project/](my_ml_project/).

---

## 📌 Features

- **Leakage-safe preprocessing pipeline:** median/mode imputation, `StandardScaler`, and `OneHotEncoder(handle_unknown='ignore')` combined via `ColumnTransformer`, fit only on the training split.
- **Domain feature engineering:** `Overall_Preparedness_Index` and `Skills_per_Project`, engineered from raw academic/interview/project fields.
- **Tuned XGBoost classifier:** hyperparameters selected via `RandomizedSearchCV` (15 candidates, 3-fold CV) — no SMOTE, no class weighting, standard 0.50 decision threshold.
- **Reproducible evaluation:** stratified 80:20 train/test split, 5-fold cross-validation, multi-seed stability checks, and permutation-importance based explainability (see [my_ml_project/evaluation/](my_ml_project/evaluation/) and [my_ml_project/results/](my_ml_project/results/)).
- **Gap analysis:** when a candidate is predicted "Not Employable", the API returns concrete, rule-based improvement suggestions across CGPA, interview score, internships, projects, certifications, GitHub presence, and communication skills.
- **FastAPI REST API + local web view:** `POST /predict` endpoint with interactive Swagger docs, plus a standalone browser UI ([my_ml_project/webapp/webview.html](my_ml_project/webapp/webview.html)) for manual testing without Postman.

---

## 📁 Project Structure

```
my_ml_project/
├── dataset/
│   ├── raw/student_career_success_dataset.xlsx   # Canonical training dataset (Kaggle, 50,000 records)
│   └── samples/                                  # Sample API request bodies for manual testing
├── src/
│   ├── train_model.py                            # Trains and saves the final tuned XGBoost pipeline
│   ├── compare_algorithms.py                     # Trains/compares LogReg, Decision Tree, Random Forest, XGBoost
│   └── main.py                                   # FastAPI application (loads models/, serves /predict)
├── models/
│   ├── best_employability_pipeline.pkl           # Final trained pipeline (produced by train_model.py)
│   └── label_encoder.pkl                         # Label encoder for the target column
├── evaluation/                                   # Audit/validation scripts (cross-validation, overfitting,
│                                                  # hyperparameter validity, multi-seed stability)
├── results/                                      # JSON output from the evaluation scripts
├── webapp/webview.html                           # Local browser UI for the /predict endpoint
├── report/                                       # Thesis book (Main-Book.docx, Report-02-UPDATED.docx)
├── documentation/PROJECT_STRUCTURE.md            # Full explanation of every folder/file and its purpose
└── requirements.txt
```

For a complete explanation of every file's purpose, see [my_ml_project/documentation/PROJECT_STRUCTURE.md](my_ml_project/documentation/PROJECT_STRUCTURE.md).

---

## 🚀 Setup & Installation

All commands below assume your terminal's working directory is `my_ml_project/`.

### 1. Create and activate a virtual environment

Requires **Python 3.11+** (scikit-learn 1.6.1 requires it).

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Training the Model

The model is trained directly from `dataset/raw/student_career_success_dataset.xlsx` — no separate cleaned CSV is used; cleaning and feature engineering happen in-pipeline at load time.

```bash
python src/train_model.py
```

This will:
1. Load and clean the dataset (drop duplicates, trim strings, drop leakage/excluded columns).
2. Engineer `Overall_Preparedness_Index` and `Skills_per_Project`.
3. Split the data (stratified 80:20, `random_state=42`).
4. Run `RandomizedSearchCV` to tune the XGBoost classifier.
5. Evaluate on the held-out test set (accuracy, ROC-AUC, confusion matrix, classification report).
6. Save the trained pipeline to `models/best_employability_pipeline.pkl` and `models/label_encoder.pkl`.

To compare XGBoost against Logistic Regression, Decision Tree, and Random Forest under the same protocol:

```bash
python src/compare_algorithms.py
```

---

## 🌐 Running the API

From `my_ml_project/`, start the FastAPI server with the app directory set to `src/`:

```bash
uvicorn main:app --reload --app-dir src
```

The server starts at [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive Swagger docs are available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

> The API loads `models/best_employability_pipeline.pkl` and `models/label_encoder.pkl` automatically — make sure you've run `train_model.py` at least once (or that these files already exist) before starting the server.

### Using the local web view

Instead of Postman, you can simply open [my_ml_project/webapp/webview.html](my_ml_project/webapp/webview.html) directly in your browser (double-click it, or File → Open) while the API server is running. Fill in the form and submit — the page calls `POST /predict` on `http://127.0.0.1:8000` and displays the verdict, probability, and any improvement gaps.

### Testing with Postman / curl

**Endpoint:** `POST http://127.0.0.1:8000/predict`
**Body:** raw JSON, matching the schema below. Sample request bodies are provided in [my_ml_project/dataset/samples/](my_ml_project/dataset/samples/) (`employable.json`, `not_emoloyable.json`, `employ_body_request.json`) — you can paste any of these directly as the Postman request body.

**Sample request:**

```json
{
  "Age": 23,
  "Gender": "Male",
  "University_Year": "Senior",
  "Major": "Computer Science",
  "Attendance_Percentage": 92.0,
  "Study_Hours_Per_Week": 25.0,
  "CGPA": 3.8,
  "Academic_Performance": "Good",
  "Programming_Skill": 9.5,
  "Projects_Completed": 8,
  "Certifications": 4,
  "Hackathons": 3,
  "LinkedIn_Profile": "Yes",
  "GitHub_Profile": "Yes",
  "English_Proficiency": "Advanced",
  "Teamwork": 9.0,
  "Communication_Skills": 8.5,
  "Problem_Solving": 9.0,
  "Interview_Score": 90.0,
  "Internships": 3,
  "Leadership_Experience": "Yes",
  "Resume_Score": 88.0
}
```

**Sample response (Employable):**

```json
{
  "status": "Success",
  "prediction": "Employable",
  "employability_probability": 0.888,
  "gaps_identified": [],
  "message": "Congratulations! The candidate meets the employability requirements."
}
```

**Sample response (Not Employable, with gap analysis)** — using `not_emoloyable.json`:

```json
{
  "status": "Success",
  "prediction": "Not Employable",
  "employability_probability": 0.1428,
  "gaps_identified": [
    "Low Interview Score (60.0/100). Target at least 85+ by practicing mock interviews.",
    "Insufficient Internships (0 completed). Aim for at least 2 internships to boost practical experience.",
    "Low Project Count (2 completed). Build at least 5+ real-world projects.",
    "Programming Skill Rating (5.0/10) needs improvement. Focus on Core DSA and Problem Solving.",
    "Low Certifications count (1). Complete 3+ industry-recognized certifications.",
    "Current CGPA is 2.9. Try to maintain a CGPA above 3.5.",
    "Missing GitHub Profile. Showcase your code repositories publicly.",
    "Communication Skill (6.5/10) should be improved for corporate rounds."
  ],
  "message": "Candidate needs improvement in identified gap areas to become employable."
}
```

> `gaps_identified` is only populated when the prediction is "Not Employable" — an "Employable" prediction always returns an empty list, since there is nothing to flag.

---

## 📈 Model Performance Summary

Evaluated on the held-out 10,000-record test set (decision threshold = 0.50, no SMOTE):

- **Accuracy:** 0.8170
- **ROC-AUC:** 0.8068
- **Top predictive features:** Resume_Score, Overall_Preparedness_Index, Programming_Skill, Internships, Problem_Solving

Full metrics, feature-importance tables, cross-validation and overfitting analysis, and algorithm comparisons are documented in the thesis report under [my_ml_project/report/](my_ml_project/report/) and the raw audit data under [my_ml_project/results/](my_ml_project/results/).

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
