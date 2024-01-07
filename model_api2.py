from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel
from typing import List
import pandas as pd

app = FastAPI()

# Load model
filename = 'classification_model.joblib'
model = load(filename)

# Features
features_covariate = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

@app.post('/detect_presence')
def classify():
    prediction = model.predict(pd.read_csv("datatraining.txt", index_col=0).sort_values(by=['date'])[features_covariate])
    res = ['Empty' if pred == 0 else 'Occupied' for pred in prediction]
    model_probs = model.predict_proba(pd.read_csv("datatraining.txt", index_col=0).sort_values(by=['date'])[features_covariate])[:, 1]
    return [{'label': label, 'occupation_probability': float(prob)} for label, prob in zip(res, model_probs)]

@app.get('/build')
def build_results():
        X_input = pd.read_csv("datatraining.txt", index_col=0).sort_values(by=['date']).to_dict(orient="records")
        return X_input

# Run the API with uvicorn (will run on http://127.0.0.1:8000)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
