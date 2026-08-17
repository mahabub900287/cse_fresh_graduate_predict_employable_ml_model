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

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Model and Label Encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# 2. ফাইল লোডিং টেস্ট
model = None
label_encoder = None

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print(f"Model successfully loaded from: {MODEL_PATH}")
    print(f"Label Encoder successfully loaded from: {ENCODER_PATH}")
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
            detail=f"Model files not found or failed to load! Checked path: {MODEL_PATH}"
        )

    try:
        input_data = data.dict()
        df = pd.DataFrame([input_data])

        # Feature Engineering
        df['Overall_Preparedness_Index'] = (df['Interview_Score'] * 0.7) + (df['Internships'] * 0.3)
        df['Skills_per_Project'] = df['Programming_Skill'] / (df['Projects_Completed'] + 1)

        # Prediction
        probs = model.predict_proba(df)[:, 1][0]
        custom_threshold = 0.35
        pred_class_index = int(probs >= custom_threshold)

        predicted_label = label_encoder.inverse_transform([pred_class_index])[0]

        # ==========================================
        # Gap Analysis / Recommendations Logic
        # ==========================================
        gaps = []

        # Not Employable checking
        if predicted_label == "Not Employable":
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

        # ==========================================
        # Return Final API Response
        # ==========================================
        return {
            "prediction": predicted_label,
            "employability_probability": round(float(probs), 4),
            "status": "Success",
            "gaps_identified": gaps if predicted_label == "Not Employable" else [],
            "message": "Here are the areas you need to improve to become employable." if predicted_label == "Not Employable" else "Congratulations! You meet the employability requirements."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))