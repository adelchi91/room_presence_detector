FROM python:3.9
# Set the working directory inside the container to /code
WORKDIR /code
# Copy the requirements.txt file from the host to the container's working directory
COPY ./requirements.txt /code/requirements.txt
# Install the Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt 
# Copy the model_API4.py file from the host to the container's working directory, renaming it to model_API.py
COPY ./model_API4.py /code/model_API.py 
# old file. This line is in fact useless
COPY ./classification_model /code/classification_model
# Copy the datatraining.txt file from the host to the container's working directory
COPY ./datatraining.txt /code/datatraining.txt
# Copy the datatest.txt file from the host to the container's working directory
COPY ./datatest.txt /code/datatest.txt
# Copy the classification_model.joblib file from the host to the container's working directory
COPY ./classification_model.joblib /code/classification_model.joblib
# Copy the model.py (training script) file from the host to the container's working directory
COPY ./model.py /code/model.py
# Specify the default command to run when the container starts
# In this case, it uses uvicorn to run the FastAPI application in model_API.py and listens
# on all available network interfaces (0.0.0.0).
CMD ["uvicorn", "model_API:app", "--host", "0.0.0.0"]
