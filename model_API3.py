# 1. Library imports
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib
import uvicorn
from fastapi import FastAPI

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


    # 5. Make a prediction based on the user-entered data
    #    Returns the predicted species with its respective probability
    def predict_presence(self, temperature, humidity, light, cotwo, humidityratio):
        data_in = pd.DataFrame([[temperature, humidity, light, cotwo, humidityratio]], columns=self.df.columns)
        prediction = self.model.predict(data_in)
        res = ['Empty' if pred == 0 else 'Occupied' for pred in prediction]
        probability = self.model.predict_proba(data_in)[:, 1]#[0]#.max()
        return res[0], probability[0]


# 2. Create app and model objects
app = FastAPI()
model = Model()

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_presence(room: RoomDetection):
    data = room.dict()
    prediction, probability = model.predict_presence(
        data["Temperature"], data["Humidity"], data["Light"], data["CO2"], data["HumidityRatio"]
    )
    return {
        'prediction': prediction,
        'probability': probability
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)