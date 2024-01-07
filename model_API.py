from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel
from typing import List
import pandas as pd

app = FastAPI()

# Load model
filename = 'classification_model.joblib'
model = load(filename)

# Open data

features_covariate = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

class PresenceRequest(BaseModel):
    # Define the structure of the request body
    X_train: List[dict] = []

def classify(model, X_train):
    prediction = model.predict(X_train)
    res = ['Empty' if pred == 0 else 'Occupied' for pred in prediction]
    model_probs = model.predict_proba(X_train)[:, 1]
    return [{'label': label, 'occupation_probability': float(prob)} for label, prob in zip(res, model_probs)]

@app.post('/detect_presence')
async def detect_presence(request: PresenceRequest):
    try:
        X_input = pd.read_csv("datatraining.txt", index_col=0)
        X_input = X_input.sort_values(by=['date'])
        X_input = df[features_covariate].to_dict(orient='records')
        predictions = classify(model, X_input)
        return {'predictions': predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API with uvicorn (will run on http://127.0.0.1:8000)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)




# from fastapi import FastAPI, HTTPException, File, UploadFile
# from joblib import load
# from pydantic import BaseModel
# from typing import List
# import pandas as pd
# import io

# app = FastAPI()

# # Load model 
# filename = 'classification_model.joblib'
# model = load(filename)

# # Define covariate and target
# features_covariate = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

# class PresenceRequest(BaseModel):
#     # Define the structure of the request body
#     X_train: List[dict]

# def classify(model, X_train):
#     # Convert the list of dictionaries back to a DataFrame
#     X_train_df = pd.DataFrame(X_train)
#     prediction = model.predict(X_train_df)
#     res = ['Empty' if pred == 0 else 'Occupied' for pred in prediction]
#     model_probs = model.predict_proba(X_train_df)[:, 1]
#     return [{'label': label, 'occupation_probability': float(prob)} for label, prob in zip(res, model_probs)]

# @app.get("/uploadfile/")
# async def upload_csv_file():
#     try:
#         # content = await file.read()
#         # df = pd.read_csv(io.StringIO(content.decode('utf-8')))
#         df = pd.read_csv("datatraining.txt", index_col=0)
#         df = df.sort_values(by=['date'])
#         X_input = df[features_covariate].to_dict(orient='records')
#         predictions = classify(model, X_input)
#         return {'predictions': predictions}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # 4. Run the API with uvicorn
# # Will run on http://127.0.0.1:8000
# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='127.0.0.1', port=8000)



# from fastapi import FastAPI, HTTPException
# from joblib import load
# from pydantic import BaseModel
# from typing import List
# import pandas as pd



# app = FastAPI()
# # Load model 
# filename = 'classification_model.joblib'
# model = load(filename)
# # Open data
# df = pd.read_csv("datatraining.txt", index_col=0)
# # sort dataframe by date
# df = df.sort_values(by=['date'])
# # define covariate and target
# features_covariate = ["Temperature",  "Humidity", "Light", "CO2", "HumidityRatio"]
# X_train = df[features_covariate].copy()


# class PresenceRequest(BaseModel):
#     # Define the structure of the request body
#     # Convert DataFrame to a list of dictionaries
#     X_train: List[dict]

# # @app.post('/predict')
# def classify(model, X_train):
#     prediction = model.predict(X_train)
#     res = ['Empty' if pred == 0 else 'Occupied' for pred in prediction]
#     model_probs = model.predict_proba(X_train)[:, 1]
#     return [{'label': label, 'occupation_probability': float(prob)} for label, prob in zip(res, model_probs)]


# # @app.get('/')
# # def get_root():
# #     return {'message': 'Welcome to the presence detection API'}

# # @app.post('/')
# # async def detect_presence(X_train):
# #     return classify(model, X_train)

# # @app.post('/detect_presence')
# # async def detect_presence(request: PresenceRequest):
# #     try:
# #         X_input = request.X_train
# #         predictions = classify(model, X_input)
# #         return {'predictions': predictions}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

# @app.post('/detect_presence')
# async def detect_presence(request: PresenceRequest):
#     try:
#         X_input = request.dict()['X_train']
#         predictions = classify(model, X_input)
#         return {'predictions': predictions}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

        

# # 4. Run the API with uvicorn
# #    Will run on http://127.0.0.1:8000
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)