# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List
# import numpy as np
# import pandas as pd
# import pickle

# app = FastAPI(title="Course Failure Risk Prediction API",
#               description="High performance API for Failure ML Model",
#               version="1.0.0")

# # ---------------- Load Preprocessor & Model Once ---------------- #
# MODEL_PATH = "artifacts/random_forest_model.pkl"
# PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

# with open(MODEL_PATH, "rb") as f:
#     model = pickle.load(f)

# with open(PREPROCESSOR_PATH, "rb") as f:
#     preprocessor = pickle.load(f)


# # ------------------- Input Schema ------------------- #
# class InputData(BaseModel):
#     Age: float
#     Gender: str
#     City: str
#     Highest_Qualification: str
#     Stream: str
#     Year_Of_Completion: int
#     Are_you_currently_working: str
#     Your_Designation: str
#     Employment_Type: str
#     First_Name: str = None
#     Last_Name: str = None
#     Company_Name: str = None


# class BatchInput(BaseModel):
#     records: List[InputData]


# # ------------------- Home ------------------- #
# @app.get("/")
# def home():
#     return {"message": "Failure Risk Prediction API Running Successfully 🚀"}


# # ------------------- Single Prediction ------------------- #
# @app.post("/predict")
# def predict(data: InputData):
#     df = pd.DataFrame([data.dict()])
#     processed = preprocessor.transform(df)
#     prediction = model.predict(processed)[0]

#     return {"Failure_Prediction": float(prediction)}


# # ------------------- Bulk Prediction (Millions Supported) ------------------- #
# @app.post("/predict_bulk")
# def predict_bulk(batch: BatchInput):
#     df = pd.DataFrame([row.dict() for row in batch.records])

#     processed = preprocessor.transform(df)

#     preds = model.predict(processed)

#     return {"total_records": len(preds),
#             "predictions": preds.tolist()}

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import pickle

app = FastAPI(
    title="Failure Risk Prediction API",
    description="High performance ML API",
    version="1.0.0"
)

# ------------ CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ Static + Templates ----------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------ Load Model + Preprocessor ----------------
MODEL_PATH = "artifacts/random_forest_model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

# ------------ Schemas ----------------
class InputData(BaseModel):
    Age: float
    Gender: str
    City: str
    Highest_Qualification: str
    Stream: str
    Year_Of_Completion: int
    Are_you_currently_working: str
    Your_Designation: str
    Employment_Type: str
    First_Name: Optional[str] = None
    Last_Name: Optional[str] = None
    Company_Name: Optional[str] = None


class BatchInput(BaseModel):
    records: List[InputData]


# ------------ UI Page ----------------
@app.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ------------ API Health ----------------
@app.get("/health")
def health():
    return {"message": "Prediction API Running Successfully 🚀"}


# ------------ Single Prediction ----------------
@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    processed = preprocessor.transform(df)
    prediction = model.predict(processed)[0]

    return {
    "status": "success",
    "prediction": str(prediction)
}



# ------------ Bulk Prediction ----------------
@app.post("/predict_bulk")
def predict_bulk(batch: BatchInput):
    df = pd.DataFrame([row.dict() for row in batch.records])
    processed = preprocessor.transform(df)
    preds = model.predict(processed)

    return {
        "total_records": len(preds),
        "predictions": preds.tolist()
    }
