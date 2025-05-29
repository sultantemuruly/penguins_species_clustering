from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, world!"}


# Load models
model_dir = os.path.join(os.path.dirname(__file__), "../model")
kmeans_pipeline = joblib.load(os.path.join(model_dir, "kmeans_pipeline.pkl"))


class InputData(BaseModel):
    culmen_length_mm: float = Field(..., example=39.1)
    culmen_depth_mm: float = Field(..., example=18.7)
    flipper_length_mm: float = Field(..., example=181)
    body_mass_g: float = Field(..., example=3750)
    sex: str = Field(..., example="MALE")


@app.post("/predict/kmeans")
def predict_kmeans(data: InputData):
    try:
        df = pd.DataFrame(
            [
                {
                    "culmen_length_mm": data.culmen_length_mm,
                    "culmen_depth_mm": data.culmen_depth_mm,
                    "flipper_length_mm": data.flipper_length_mm,
                    "body_mass_g": data.body_mass_g,
                    "sex": data.sex,
                }
            ]
        )
        label = kmeans_pipeline.predict(df)[0]
        return {"model": "kmeans", "label": int(label)}
    except Exception as e:
        return {"error": str(e)}
