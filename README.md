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



📜 Full Process & ML Methodology1. Data Cleaning & PreprocessingDuplicate Removal: df.drop_duplicates() applied to ensure record uniqueness.String Trimming: str.strip() applied across all string columns to resolve formatting mismatches.Preprocessing Pipeline: Combined via ColumnTransformer:Numerical Pipeline: SimpleImputer(strategy='median') ➔ StandardScaler()Categorical Pipeline: SimpleImputer(strategy='most_frequent') ➔ OneHotEncoder(handle_unknown='ignore')2. Custom Feature EngineeringOverall_Preparedness_Index: Synthesizes overall student readiness combining weighted interview and practical metrics:$$\text{Overall\_Preparedness\_Index} = (\text{Interview\_Score} \times 0.7) + (\text{Internships} \times 0.3)$$Skills_per_Project: Evaluates programming efficiency per completed practical project:$$\text{Skills\_per\_Project} = \frac{\text{Programming\_Skill}}{\text{Projects\_Completed} + 1}$$3. Model Training & Imbalance StrategySplit Strategy: Stratified 80/20 train-test split (stratify=y).Pipeline Integration: imbalanced-learn's ImbPipeline wraps preprocessing, SMOTE oversampling, and classification to avoid leakage during cross-validation.XGBoost Classifier: Hyper-parameters configured with n_estimators=200, learning_rate=0.05, max_depth=6, and tuned scale_pos_weight.4. Custom Decision Boundary & Feature ImportanceThreshold Selection: Evaluated probability distributions to set an optimal threshold at 0.35 for maximized F1-score and Recall.Feature Importance: Evaluated using permutation_importance to identify top predictive drivers powering the gap analysis recommendations.📁 Repository & Project StructurePlaintext.
├── best_employability_pipeline.pkl      # Serialized Scikit-Learn + SMOTE + XGBoost pipeline
├── label_encoder.pkl                    # Serialized LabelEncoder artifact
├── main.py                              # FastAPI backend application & endpoint logic
├── student_career_success_dataset.csv   # Dataset used for model training
├── requirements.txt                     # Project dependencies
└── README.md                            # Complete Project Documentation
📊 Dataset Features & SchemaFeature NameTypeDescriptionAgeIntegerStudent age in yearsGenderCategoricalMale / FemaleCGPAFloatCumulative Grade Point Average (0.00 - 4.00)InternshipsIntegerTotal completed internshipsProjects_CompletedIntegerTotal software/engineering projects builtProgramming_SkillFloatSelf/Assessed skill score (1.0 - 10.0)Interview_ScoreFloatMock/Real interview evaluation score (0.0 - 100.0)Communication_SkillsFloatSoft skill rating (1.0 - 10.0)CertificationsIntegerProfessional certifications completedGitHub_ProfileCategoricalYes / NoEmployability_StatusCategorical (Target)Employable / Not Employable🛠️ Tech Stack & DependenciesProgramming Language: Python 3.10+Machine Learning & Analytics: scikit-learn, xgboost, imbalanced-learn, pandas, numpy, joblibAPI Framework: FastAPI, Uvicorn, pydanticVisualization (Notebook): matplotlib, seaborn🚀 Setup & Installation Guide1. Clone the RepositoryBashgit clone https://github.com/your-username/student-employability-system.git
cd student-employability-system
2. Set Up Virtual EnvironmentBash# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
3. Install DependenciesBashpip install -r requirements.txt
4. Verify Model ArtifactsEnsure that best_employability_pipeline.pkl and label_encoder.pkl are located in the root directory alongside main.py.5. Run the FastAPI ServerBashuvicorn main:app --reload
The server will start locally at [http://127.0.0.1:8000](http://127.0.0.1:8000).🧪 API Endpoints & TestingAccess the interactive Swagger UI at:👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)Endpoint: POST /predictSample Input Payload (JSON):JSON{
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
Sample Response 1: Not Employable (With Gap Analysis)JSON{
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
Sample Response 2: EmployableJSON{
  "status": "Success",
  "prediction": "Employable",
  "employability_probability": 0.8120,
  "gaps_identified": [],
  "message": "Congratulations! You meet the employability requirements."
}
📈 Model Evaluation SummaryPrimary Classification Threshold: 0.35Core Evaluation Metrics: High Recall and F1-score achieved on minority class (Employable).Key Determinants: Interview Score, Overall Preparedness Index, CGPA, and Programming Skill.📤 GitHub Deployment Quick CommandsTo push this complete project to GitHub, run the following commands in your terminal:Bashgit init
git add .
git commit -m "feat: complete machine learning pipeline and FastAPI backend integration"
git branch -M main
git remote add origin https://github.com/your-username/student-employability-system.git
git push -u origin main
📜 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.