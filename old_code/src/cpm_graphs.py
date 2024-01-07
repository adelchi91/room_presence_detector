import os
import sys

import joblib
import numpy as np
import yaml
from pandas import concat, merge, options

from helpers_local.helpers_factorizing import (
    calibration_by_feature,
    out_of_time_split,
    plot_cpm_graphs,
)
from helpers_local.helpers_model_2_1 import v2_features  # NOQA
from pipeline_files.helpers import v3_cat_features, v3_features  # NOQA

options.display.max_columns = 40
options.display.width = 1000

if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython cpm_graphs.py training_dataset_path v3_1_model_path and v2_1_model_path")
    sys.exit(1)

# Load files
training_dataset_path = sys.argv[1]
model_3_1_path = sys.argv[2]
model_2_1_path = sys.argv[3]

# Load files
df = joblib.load(open(training_dataset_path, "rb"))
df.rename(
    columns={"remainder__contract_reference": "contract_reference", "remainder__application_date": "application_date"},
    inplace=True,
)
# Load params
params = yaml.safe_load(open("params.yaml"))
target = params["target"]
test_size = params["validation_size"]
apply_filter = params["cpm_graphs"]["apply_filter"]
apply_decile = params["cpm_graphs"]["apply_decile"]


# Out of time split
X, y, X_train, X_val, y_train, y_val = out_of_time_split(df, target, test_size)
model_3_1 = joblib.load(open(model_3_1_path, "rb"))
model_2_1 = joblib.load(open(model_2_1_path, "rb"))

models = [model_2_1, model_3_1]
for i, model_chosen in enumerate(models):
    if i == 0:
        model_name = "model_2_1"
    else:
        model_name = "model_3_1"
    # predict proba
    train_predictions = model_chosen.predict_proba(X_train.drop(columns=[target]))[:, 1]
    val_predictions = model_chosen.predict_proba(X_val.drop(columns=[target]))[:, 1]

    new_version = "2"  # Number version of the new candidate model
    df_train = X_train.assign(dataframe="train_test", predict_prob=train_predictions)
    df_val = X_val.assign(dataframe="validation", predict_prob=val_predictions)
    df_full = concat([df_train, df_val], axis=0)
    df_full_transformed = (
        model_3_1.named_steps["preprocess"]
        .transform(df_full)
        .rename(
            columns={
                "remainder__contract_reference": "contract_reference",
                "remainder__application_date": "application_date",
            }
        )
    )
    df_full_transformed = merge(
        df_full_transformed,
        df_full[["contract_reference", "predict_prob", target]],
        on="contract_reference",
        how="left",
    ).sort_values("application_date")
    if apply_filter:
        if apply_decile:
            # Analysis with ISO preacc
            # Calculate the 70th percentile (7th decile)
            decile_7 = df_full_transformed["predict_prob"].quantile(0.7)

            # Select values within the first 7 deciles
            df_full_transformed = df_full_transformed[df_full_transformed["predict_prob"] <= decile_7]
        else:
            # analysis at DIFFERENT preacc
            df_full_transformed = df_full_transformed[lambda x: x.predict_prob < 0.1]

    def convert_dataframe(alt_w_pred, vars, vars_cat):
        for var in vars:
            alt_w_pred[var] = alt_w_pred[var].astype(np.float32)
        for var in vars_cat:
            alt_w_pred[var] = alt_w_pred[var].astype("string")
        return alt_w_pred

    numerical_vars = list(
        set(v3_features) - set(v3_cat_features)
    )  # ['main_net_monthly_income','mortgage_amount', 'personal_age']
    categorical_vars = v3_cat_features + [
        "remainder__business_provider_code"
    ]  # ['verified_is_homeowner', 'verified_bank_code', 'business_provider_code', 'verified_housing_code']

    # table calibration
    cpm_vars = [
        "simpleimputer-1__bank_age",
        "remainder__bank_code",
        "remainder__marital_status_code",
        "remainder__business_provider_code",
        "remainder__housing_code",
        "remainder__main_net_monthly_income",
        "remainder__marital_status_code",
        "simpleimputer-2__mortgage_amount",
        "simpleimputer-1__personal_age",
    ]
    table_outputs = []
    for var in cpm_vars:
        output = calibration_by_feature(
            df_full_transformed, feature_name=var, prediction_name="predict_prob", event_name=target
        )
        table_outputs.append(output)

    df_calibration_table = concat(table_outputs, axis=0)
    os.makedirs(os.path.join("data", "cpm_graphs"), exist_ok=True)
    df_calibration_table.to_csv(
        os.path.join(
            "data", "cpm_graphs", f"variables_calibration_{model_name}_filters_{apply_filter}_{apply_decile}.csv"
        )
    )

    # CPM graphs
    df_full_converted = convert_dataframe(df_full_transformed, numerical_vars, categorical_vars)
    df_full_converted[target] = df_full_converted[target].astype("int64")
    plot_cpm_graphs(df_full_converted, df_full_converted[target], new_version, model_name, apply_filter, apply_decile)
    print("Hi")
