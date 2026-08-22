import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Student Placement Prediction API",
    description="API to predict student placement status using XGBoost model.",
    version="1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
        input_data = data.dict()
        df = pd.DataFrame([input_data])

        # Feature Engineering
        df['Overall_Preparedness_Index'] = (df['Interview_Score'] * 0.7) + (df['Internships'] * 0.3)
        df['Skills_per_Project'] = df['Programming_Skill'] / (df['Projects_Completed'] + 1)

        # 1. Label Encoder index identification
        # Check which index represents 'Employable'
        classes_list = list(label_encoder.classes_)
        employable_idx = classes_list.index("Employable") if "Employable" in classes_list else 0

        # 2. Get Probability for 'Employable'
        probabilities = model.predict_proba(df)[0]
        employable_prob = probabilities[employable_idx]

        # 3. Decision Boundary threshold check
        custom_threshold = 0.35
        if employable_prob >= custom_threshold:
            predicted_label = "Employable"
        else:
            predicted_label = "Not Employable"

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
            # Employable JSON Structure
            return {
                "status": "Success",
                "prediction": "Employable",
                "employability_probability": round(float(employable_prob), 4),
                "gaps_identified": [],
                "message": "Congratulations! The candidate meets the employability requirements."
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))