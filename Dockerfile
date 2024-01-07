FROM python:3.9
# 
WORKDIR /code
# 
COPY ./requirements.txt /code/requirements.txt
# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt 
# 
COPY ./model_API4.py /code/model_API.py 
#
COPY ./classification_model /code/classification_model
#
COPY ./datatraining.txt /code/datatraining.txt
#
COPY ./classification_model.joblib /code/classification_model.joblib
#
COPY ./model.py /code/model.py
#
CMD ["uvicorn", "model_API:app", "--host", "0.0.0.0"]
