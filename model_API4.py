# 1. Library imports
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib
import uvicorn
from fastapi import FastAPI
from typing import List
from fastapi import HTTPException



FEATURES_COVARIATE = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]



# 2. Class which describes a single flower measurements
class RoomDetection(BaseModel):
    Temperature: float 
    Humidity: float 
    Light: float 
    CO2: float
    HumidityRatio: float


# 3. Class for training the model and making predictions
class Model:
    # 6. Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and 
    #    saves the model
    def __init__(self):
        self.df = pd.read_csv('datatraining.txt')[FEATURES_COVARIATE]
        self.model_fname_ = 'classification_model.joblib'
        self.model = joblib.load(self.model_fname_)

    # # 5. Make a prediction based on the user-entered data
    # #    Returns the predicted species with its respective probability
    def predict_presence(self, data: List[RoomDetection]):
        input_data = pd.DataFrame([[room.Temperature, room.Humidity, room.Light, room.CO2, room.HumidityRatio] for room in data], columns=self.df.columns)
        predictions = self.model.predict(input_data)
        results = [{'prediction': 'Empty' if pred == 0 else 'Occupied',
                    'probability': prob} for pred, prob in zip(predictions, self.model.predict_proba(input_data)[:, 1])]
        return results


# 2. Create app and model objects
app = FastAPI()
model = Model()

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_presence(rooms: List[RoomDetection]):
    if not rooms:
        raise HTTPException(status_code=400, detail="Input list is empty")
    return model.predict_presence(rooms)


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
#if __name__ == '__main__':
#    uvicorn.run(app, host='127.1.1.1', port=8080)
