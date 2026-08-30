# Student Placement Prediction API

This is a Machine Learning-powered REST API built with **FastAPI** and **XGBoost** to predict student career placement outcomes based on academic performance, technical skills, and extra-curricular activities.

--- 

## 🛠️ Tech Stack & Dependencies

The project relies on the following core tools and libraries:

### Framework & API Tools
* **Python 3.12+**
* **FastAPI** (`0.115.0+`): Web framework for building APIs.
* **Uvicorn** (`0.30.0+`): ASGI server to run the FastAPI application.
* **Pydantic**: Data validation and settings management.

### Machine Learning & Data Processing
* **scikit-learn** (`1.6.1`): Preprocessing pipelines, standard scaling, and one-hot encoding.
  > ⚠️ **Note:** Version `1.6.1` is strictly required to ensure compatibility with saved pickle artifacts.
* **XGBoost**: Gradient boosting framework used for model inference.
* **imbalanced-learn** (`0.12.0+`): For handling imbalanced datasets using SMOTE inside pipelines.
* **pandas**: Data manipulation and DataFrame structure formatting.
* **numpy**: Numerical calculations and array handling.
* **joblib**: Model serialization and loading `.pkl` files.

---

## 📋 Requirements File (`requirements.txt`)

Create a `requirements.txt` file in your root folder and add the following dependencies:

```text
fastapi
uvicorn
pydantic
scikit-learn==1.6.1
xgboost
imbalanced-learn
pandas
numpy
joblib