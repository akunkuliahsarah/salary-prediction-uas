from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("salary_prediction_model.pkl")

app = FastAPI(title="Salary Prediction API")

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class YearsData(BaseModel):
    YearsExperience: float

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict")
def predict_salary(data: YearsData):
    years = np.array([[data.YearsExperience]])  
    salary = model.predict(years)[0]
    return {
        "YearsExperience": data.YearsExperience,
        "PredictedSalary": round(float(salary), 2)
    }