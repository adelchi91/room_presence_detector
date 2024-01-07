import pathlib
import sys

import joblib
import pandas as pd
import sklearn.model_selection
import yaml

from utils import compute_global_figures, pprint_dataframe

# isort: off
sys.path.append("./src/pipeline_files")
from helpers import (  # NOQA
    preprocess_pipeline_model_3p1 as preprocess_pipeline_model_3p1,
)

# isort: on
sklearn.set_config(transform_output="pandas")

""""
The logic in the preprocessing step is to retrieve the data, which underwent the quality controls with the pydantic
model and then create features. No model fitting is carried out until we reach the following cross-validation step.
X-covariate data is all the data coming from the datamart that is controlled previously by the Pydantic model.
y-data contains the defaults from the risk table, as well as any other variable that is used for other purposes (some
might come form the datamart), that do not need to be passed in the Pydantic model.
"""

# Loading paramters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
# target
target = params["target"]
# path definition for output data
data = pathlib.Path("data")

# Dataframes imports
# all data but defaults
X = pd.read_pickle(data / "payloads_dataset.pkl")
# Creating features for model candidate V3.1 and recreating features of model in production, i.e. V2.1
X = preprocess_pipeline_model_3p1.fit_transform(X)
# removing contracts with missing "borrowed_amount"
# X = X.dropna(subset=["borrowed_amount"])
# defaults
y = pd.read_pickle(data / "targets_dataset.pkl")

# DEMANDS
X.rename(
    columns={"remainder__contract_reference": "contract_reference", "remainder__application_date": "application_date"},
    inplace=True,
)
X = pd.merge(X, y, on="contract_reference", how="left").sort_values("application_date")
print("Figures for the whole dataset")
pprint_dataframe(compute_global_figures(X, target).hide(axis="index"))

preprocess = data / "preprocess"
preprocess.mkdir(exist_ok=True)

# Printing simple stats on datafame demands
X_stats_demands = compute_global_figures(X, target).data

# Saving data
X_stats_demands.to_csv(preprocess / "X_stats_demands.csv")
joblib.dump(X, preprocess / "X.joblib")
joblib.dump(preprocess_pipeline_model_3p1, preprocess / "pipeline.joblib")

# FINANCED
X = X.query("decision_status == 'granted'")
print("Figures for the dataset filtered on financed population")
pprint_dataframe(compute_global_figures(X, target).hide(axis="index"))

# Printing simple stats on datafame financed
X_stats_financed = compute_global_figures(X, target).data
# Saving data
X_stats_financed.to_csv(preprocess / "X_stats_financed.csv")

# Saving data
gmv = preprocess / "gmv"
gmv.mkdir(exist_ok=True)
joblib.dump(X, gmv / "X.joblib")
