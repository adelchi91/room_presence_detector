import os
import sys

import joblib
import pandas as pd
import yaml
from pandas import options

"""
Creation of temporal calibration plot. We want to identify whether the new candidate model is well-calibrated time-wise.
A second plot with the calibration of each model built for each split, during the cross-validation, is shown to
underline the effect of adding data at each temporal split and how it impacts the model calibration.
"""

options.display.max_columns = 40
options.display.width = 1000

# verification that the correct number of arguments is being fed at this stage. The arguments are defined in dvc.yaml
# file.
if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython stats_model.py training_dataset_path validation_dataset_path pipeline_path variables_file"
    )
    sys.exit(1)

# Load files
payloads_dataset_path = sys.argv[1]
targets_dataset_path = sys.argv[2]
model_3p1_path = sys.argv[3]


# Load files
# Dataframes of interest
X = pd.read_pickle(payloads_dataset_path)
X.rename(
    columns={"remainder__contract_reference": "contract_reference", "remainder__application_date": "application_date"},
    inplace=True,
)
# defaults
y = pd.read_pickle(targets_dataset_path)
# model 3.1 pipeline
model_3p1 = joblib.load(open(model_3p1_path, "rb"))

# Load params
params = yaml.safe_load(open("params.yaml"))
target = params["target"]
test_size = params["validation_size"]

# retrieving the contract reference column needed to merge X and y, given that the request_id in risk table
# are inaccurate
X_expanded = pd.json_normalize(X["payload"], meta=["request_id"])
X["contract_reference"] = X_expanded["contract_reference"].copy()
X["application_date"] = pd.to_datetime(X_expanded["application_date"]).copy()
# Delete the DataFrame to free up memory
del X_expanded
# Merging
X = pd.merge(X, y, on="contract_reference", how="left").sort_values("application_date")
# Copy of X for cpm replay
X_full = X.copy()
# perimeter of interest where the target is defined
X = X.dropna(subset=(target,))
y = X[target].astype(float)

# CPM replay
X_replay = X_full.copy()
X_full["prediction_v3"] = model_3p1.predict_proba(X_full)[:, 1]
X_replay = (
    model_3p1.named_steps["preprocess"]
    .transform(X_full)
    .rename(
        columns={
            "remainder__contract_reference": "contract_reference",
            "remainder__application_date": "application_date",
        }
    )
)
X_replay = pd.merge(
    X_replay, X_full[["contract_reference", "prediction_v3", target]], on="contract_reference", how="left"
).sort_values("application_date")

# Save outputs
os.makedirs(os.path.join("data", "dataframe_for_analytics"), exist_ok=True)
X_full.to_pickle(os.path.join("data", "dataframe_for_analytics", "df_full.pkl"))
X_replay.to_pickle(os.path.join("data", "dataframe_for_analytics", "df_replay.pkl"))
joblib.dump(X, os.path.join("data", "dataframe_for_analytics", "X.joblib"))
