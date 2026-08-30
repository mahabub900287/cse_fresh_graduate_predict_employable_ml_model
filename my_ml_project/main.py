import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Student Placement Prediction API",
    description="API to predict student placement status using the tuned XGBoost model.",
    version="2.0"
)

# Allows the local webview.html (opened via file://) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Retrained pipeline files (no SMOTE, tuned hyperparameters via RandomizedSearchCV)
MODEL_PATH = os.path.join(BASE_DIR, 'best_employability_pipeline.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

model = None
label_encoder = None

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Model and Label Encoder loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR during loading: {e}")


class StudentData(BaseModel):
    Age: int
    Gender: str
    University_Year: str
    Major: str
    Attendance_Percentage: float
    Study_Hours_Per_Week: float
    CGPA: float
    Academic_Performance: str
    Programming_Skill: float
    Projects_Completed: int
    Certifications: int
    Hackathons: int
    LinkedIn_Profile: str
    GitHub_Profile: str
    English_Proficiency: str
    Teamwork: float
    Communication_Skills: float
    Problem_Solving: float
    Interview_Score: float
    Internships: int
    Leadership_Experience: str
    Resume_Score: float


@app.get("/")
def read_root():
    return {"message": "Welcome to Student Placement Prediction API!"}


@app.post("/predict")
def predict_placement(data: StudentData):
    if model is None or label_encoder is None:
        raise HTTPException(
            status_code=500,
            detail="Model files not found or failed to load!"
        )

    try:
        # CHANGED: .dict() is deprecated in Pydantic v2 -> use .model_dump()
        input_data = data.model_dump()
        df = pd.DataFrame([input_data])

        # Feature Engineering (unchanged - matches training pipeline)
        df['Overall_Preparedness_Index'] = (df['Interview_Score'] * 0.7) + (df['Internships'] * 0.3)
        df['Skills_per_Project'] = df['Programming_Skill'] / (df['Projects_Completed'] + 1)

        # NOTE: the trained pipeline (v3) was fit on 18 features and excludes
        # Age, Gender, University_Year, Major, Attendance_Percentage and
        # LinkedIn_Profile. The ColumnTransformer selects columns by name and
        # will simply ignore these extra columns, so they can safely stay in
        # the request schema without affecting the prediction.

        # 1. Label Encoder index identification
        classes_list = list(label_encoder.classes_)
        # CHANGED: we now need the "Not Employable" (positive/risk) index,
        # not the "Employable" index, because the threshold below is applied
        # to the risk class -- applying it to the wrong class silently
        # inverted the decision rule in the previous version of this file.
        not_employable_idx = (
            classes_list.index("Not Employable")
            if "Not Employable" in classes_list else 1
        )

        # 2. Get probability of the risk class ("Not Employable")
        probabilities = model.predict_proba(df)[0]
        not_employable_prob = probabilities[not_employable_idx]
        employable_prob = 1 - not_employable_prob

        # 3. Decision Boundary threshold check
        # CHANGED: threshold reverted to the standard 0.50 default, matching
        # the tuned training pipeline, which no longer uses SMOTE / a
        # lowered custom threshold (see Section 5.6/5.7 of the report for
        # why that combination was removed). The comparison is now applied
        # to the correct (Not Employable) probability.
        custom_threshold = 0.50
        if not_employable_prob >= custom_threshold:
            predicted_label = "Not Employable"
        else:
            predicted_label = "Employable"

        # ==========================================
        # Response Formatting & Gap Analysis
        # ==========================================
        if predicted_label == "Not Employable":
            gaps = []
            if data.Interview_Score < 85.0:
                gaps.append(f"Low Interview Score ({data.Interview_Score}/100). Target at least 85+ by practicing mock interviews.")

            if data.Internships < 2:
                gaps.append(f"Insufficient Internships ({data.Internships} completed). Aim for at least 2 internships to boost practical experience.")

            if data.Projects_Completed < 5:
                gaps.append(f"Low Project Count ({data.Projects_Completed} completed). Build at least 5+ real-world projects.")

            if data.Programming_Skill < 8.0:
                gaps.append(f"Programming Skill Rating ({data.Programming_Skill}/10) needs improvement. Focus on Core DSA and Problem Solving.")

            if data.Certifications < 3:
                gaps.append(f"Low Certifications count ({data.Certifications}). Complete 3+ industry-recognized certifications.")

            if data.CGPA < 3.5:
                gaps.append(f"Current CGPA is {data.CGPA}. Try to maintain a CGPA above 3.5.")

            if data.GitHub_Profile.lower() in ["no", "false"]:
                gaps.append("Missing GitHub Profile. Showcase your code repositories publicly.")

            if data.Communication_Skills < 8.0:
                gaps.append(f"Communication Skill ({data.Communication_Skills}/10) should be improved for corporate rounds.")

            return {
                "status": "Success",
                "prediction": "Not Employable",
                "employability_probability": round(float(employable_prob), 4),
                "gaps_identified": gaps,
                "message": "Candidate needs improvement in identified gap areas to become employable."
            }

        else:
            return {
                "status": "Success",
                "prediction": "Employable",
                "employability_probability": round(float(employable_prob), 4),
                "gaps_identified": [],
                "message": "Congratulations! The candidate meets the employability requirements."
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
