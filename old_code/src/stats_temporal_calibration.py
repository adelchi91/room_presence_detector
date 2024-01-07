import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from pandas import options
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from yc_younipy.preprocessing.quantile_binning import QuantileBinning

"""
Creation of temporal calibration plot. We want to identify whether the new candidate model is well-calibrated time-wise.
A second plot with the calibration of each model built for each split, during the cross-validation, is shown to
 underline the effect of adding data at each temporal split and how it impacts the model calibration.
"""

options.display.max_columns = 40
options.display.width = 1000

# verification that the correct number of arguments is being fed at this stage. The arguments are defined in dvc.yaml
# file.
if len(sys.argv) != 6:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython stats_model.py training_dataset_path validation_dataset_path pipeline_path variables_file"
    )
    sys.exit(1)

# Load files
model_3_1_path = sys.argv[1]
model_2_1_path = sys.argv[2]
split_end_dates_path = sys.argv[3]
split_end_dates_train_path = sys.argv[4]
payloads_dataset_path = sys.argv[5]

# Load params
params = yaml.safe_load(open("params.yaml"))
target = params["target"]
test_size = params["validation_size"]

# Load files
# Dataframes of interest
# X = joblib.load(payloads_dataset_path)
X = pd.read_pickle(payloads_dataset_path)
y = X[target].astype(float)
model_3_1 = joblib.load(open(model_3_1_path, "rb"))
model_2_1 = joblib.load(open(model_2_1_path, "rb"))
# threshold end-date for test splits
split_end_dates = pd.to_datetime(joblib.load(open(split_end_dates_path, "rb")))
# threshold end-date for train splits
split_end_dates_train = pd.to_datetime(joblib.load(open(split_end_dates_train_path, "rb")))


# retrieving the contract reference column needed to merge X and y, given that the request_id in risk table
# are inaccurate
# X_expanded = pd.json_normalize(X["payload"], meta=['request_id'])
# X["contract_reference"] = X_expanded["contract_reference"].copy()
# X["application_date"] = pd.to_datetime(X_expanded["application_date"]).copy()
# # Delete the DataFrame to free up memory
# del X_expanded
# # Merging
# X = pd.merge(X, y, on="contract_reference", how="left").sort_values("application_date")
# # perimeter of interest where the target is defined
# X = X.dropna(subset=(target,))
# y = X[target].astype(float)
# We need to select only payload and request id to send the data in the pipeline.
# X = X[['request_id', 'payload']]

# predict proba
predictions = model_3_1.predict_proba(X[["request_id", "payload"]])[:, 1]
# complete dataframe including predictions
df_full = X.copy()
df_full = df_full.assign(predict_prob=predictions, predict_score=(1 - predictions) * 10000)
# Formatting dataframes for the relevant plots
data_replay_piv = df_full.melt(
    id_vars=["contract_reference", "predict_score"],
    value_vars=["dn2_6", "dn3_12"],
    var_name="indicator_type",
    value_name="indicator_value",
).melt(
    id_vars=["contract_reference", "indicator_type", "indicator_value"],
    value_vars=["predict_score"],
    var_name="score_type",
    value_name="score_value",
)


def plot_temporal_calibration(df_target_score, split_end_dates):
    """
    Temporal calibration all over time domain for the best model. Split test end dates are shown with vertical black
     lines
    @param df_target_score: dataframe of interest to plot
    @param split_end_dates: test end dates - list
    @return:
    """
    # separate the two plots, use point plot and do a boostrap to add error bars
    fig, ax = plt.subplots()
    # Set seaborn style
    sns.set_style("whitegrid")

    # Create plot
    sns.lineplot(data=df_target_score, x="cohort", y="avg_risk_rate", ci=95, ax=ax, color="blue", label="Target Score")

    sns.lineplot(
        data=df_predict_score, x="cohort", y="avg_risk_rate", ci=None, ax=ax, color="red", label="Predict Score"
    )

    #
    # Add grid
    ax.grid(True, linestyle="--", axis="y", color="gray", alpha=0.5)
    # plt.axvline(x=split_date, color='black', linestyle='--')

    # Iterate over split_end_dates and add vertical lines
    for ii, date in enumerate(split_end_dates):
        plt.axvline(x=date, color="black", linestyle="--", label=f"Splits_{ii + 1}")

    # Set plot title and axis labels
    ax.set_title("Average Risk Rate by Cohort and Score Type")
    ax.set_xlabel("Dates")
    ax.set_ylabel("Average Risk Rate")
    ax.set_ylim(0, 0.25)

    # Adjust legend position
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)

    # Add tight layout
    plt.tight_layout()
    plt.savefig(
        os.path.join("dvc_plots", "temporal_calibration", "temporal_calibration.png"), dpi=600, bbox_inches="tight"
    )
    # Show plot
    plt.show()
    return


def create_predict_score(df):
    df = df.assign(
        cohort=lambda x: pd.to_datetime(x["application_date"]).dt.to_period("Q").dt.to_timestamp(),
        score_type="predict_score",
        score_value=lambda x: x["predict_score"],
        avg_score_value=lambda x: x["predict_score"],
        avg_risk_rate=lambda x: 1 - x["predict_score"] / 10000,
    )[["contract_reference", "cohort", "score_type", "score_value", "avg_score_value", "avg_risk_rate"]]
    return df


def create_predict_score_all_models(df):
    df = df.assign(
        cohort=lambda x: pd.to_datetime(x["application_date"]).dt.to_period("Q").dt.to_timestamp(),
        score_type="predict_score",
        avg_score_value_0=lambda x: x["predict_prob_0"],
        avg_score_value_1=lambda x: x["predict_prob_1"],
        avg_score_value_2=lambda x: x["predict_prob_2"],
        avg_score_value_3=lambda x: x["predict_prob_3"],
        avg_score_value_4=lambda x: x["predict_prob_4"],
    )[
        [
            "contract_reference",
            "cohort",
            "score_type",
            "avg_score_value_0",
            "avg_score_value_1",
            "avg_score_value_2",
            "avg_score_value_3",
            "avg_score_value_4",
        ]
    ]
    return df


def create_target_score(df):
    df = df.assign(
        cohort=lambda x: pd.to_datetime(x["application_date"]).dt.to_period("Q").dt.to_timestamp(),
        score_type="target_score",
        score_value=lambda x: 10000 * (1 - x["dn3_12"]),
        avg_score_value=lambda x: 10000 * (1 - x["dn3_12"]),
        avg_risk_rate=lambda x: 1 - 10000 * (1 - x["dn3_12"]) / 10000,
    )[["contract_reference", "cohort", "score_type", "score_value", "avg_score_value", "avg_risk_rate"]]
    return df


def retrieve_models_for_each_split(cv, X_train, y_train):
    """
    Recreates the model computed on each split during the cross-validation
    @param cv: cross-validation object
    @param X_train: Covariate data
    @param y_train: target data
    @return: list of models
    """
    # Get the TimeSeriesSplit object
    tss = cv.cv
    # Initialize list to store models
    models = []
    # clf = cv.best_estimator_.named_steps["clf"]
    # Iterate through train/test indices
    for idx, (train_idx, test_idx) in enumerate(tss.split(X_train)):
        # Slice training data
        X_train_split = X_train.iloc[train_idx]
        y_train_split = y_train.iloc[train_idx]
        # Fit model on split data
        model = clone(cv.best_estimator_)
        model.fit(X_train_split, y_train_split)
        # Store model
        models.append(model)
    return models


def agg_fun(x):
    return pd.DataFrame.from_records(
        {
            "count": [len(x)],
            "mean_indicator_rate": [np.mean(x["indicator_value"])],
            "n_default": [np.sum(x["indicator_value"])],
            "gini": [2 * roc_auc_score(-x["indicator_value"], x["score_value"]) - 1],
        }
    )


def full_temporal_plot(df, df_all_models, name="temporal_calibration_all_models", add_all_models=True):
    fig, ax = plt.subplots()
    # Set seaborn style
    sns.set_style("whitegrid")

    # Create plot
    if add_all_models:
        legend = "Target Score"
    else:
        legend = "dn3_12"
    sns.lineplot(data=df, x="cohort", y="avg_risk_rate", ci=95, ax=ax, color="blue", label=legend)

    if add_all_models:
        for predict_probb in df_all_models.filter(regex="avg_score_value_").columns.to_list():
            sns.lineplot(data=df_all_models, x="cohort", y=predict_probb, ci=None, ax=ax, label=predict_probb)
    # Add grid
    ax.grid(True, linestyle="--", axis="y", color="gray", alpha=0.5)
    # plt.axvline(x=split_date, color='black', linestyle='--')

    # Iterate over split_end_dates and add vertical lines
    if add_all_models:
        for ii, date in enumerate(split_end_dates):
            plt.axvline(x=date, color="black", linestyle="--", label=f"Splits_{ii + 1}")

    # Set plot title and axis labels
    if add_all_models:
        ax.set_title("Average Risk Rate by Cohort and Score Type")

    ax.set_xlabel("Dates")
    ax.set_ylabel("Average Risk Rate")
    if add_all_models:
        ax.set_ylim(0, 0.25)

    # Adjust legend position
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)

    # Add tight layout
    plt.tight_layout()
    plt.savefig(os.path.join("dvc_plots", "temporal_calibration", f"{name}.png"), dpi=600, bbox_inches="tight")
    # Show plot
    plt.show()
    return


# Refomatting dataframes
data_replay_calib = (
    data_replay_piv.assign(
        score_bands=lambda x: QuantileBinning(nb_bins=6, output_labels=True).fit_transform(x.loc[:, ["score_value"]]),
        pred_value=lambda x: 1 - x["score_value"] / 10000,
    )
    .groupby(["score_bands", "indicator_type", "score_type"], as_index=False)
    .agg(
        count=("contract_reference", "count"),
        mean_indicator_value=("indicator_value", "mean"),
        mean_pred_value=("pred_value", "mean"),
    )
)

data_replay_avg_score = (
    df_full.assign(
        cohort=lambda x: pd.to_datetime(x["application_date"]).dt.to_period("Q").dt.to_timestamp(),
        target_score=lambda x: 10000 * (1 - x["dn3_12"]),
    )
    .melt(
        id_vars=["contract_reference", "cohort"],
        value_vars=["predict_score", "target_score"],
        var_name="score_type",
        value_name="score_value",
    )
    .groupby(["cohort", "score_type"], as_index=False)
    .agg(avg_score_value=("score_value", "mean"))
    .assign(avg_risk_rate=lambda x: 1 - x["avg_score_value"] / 10000)
)

df_predict_score = create_predict_score(df_full)

df_target_score = create_target_score(df_full)

##################
#   Saving data  #
##################
with open("./data/cross_validation/cv.joblib", "rb") as fd:
    cv = joblib.load(fd)
models = retrieve_models_for_each_split(
    cv, X.dropna(subset=(target,)), X.dropna(subset=(target,))[target].astype(float)
)
for iterator, model in enumerate(models):
    df_full[f"predict_prob_{iterator}"] = models[iterator].predict_proba(df_full.drop(columns=[target]))[:, 1]

# Save outputs
os.makedirs(os.path.join("dvc_plots", "temporal_calibration"), exist_ok=True)

#########################
# DN3_12 temporal plot ##
#########################
full_temporal_plot(df_target_score, df_target_score, name=f"{target}_default_temporal_plot", add_all_models=False)

##############################################
# temporal calibration on all training data ##
##############################################
plot_temporal_calibration(df_target_score, split_end_dates)
# We want to plot only the data after the first split
df_predict_score_all_models = create_predict_score_all_models(
    df_full[lambda x: x.application_date > split_end_dates_train[0]]
)
df_target_score = create_target_score(df_full[lambda x: x.application_date > split_end_dates_train[0]])
# plot
full_temporal_plot(
    df_target_score, df_predict_score_all_models, name="temporal_calibration_all_models", add_all_models=True
)
#
# fig, ax = plt.subplots()
# # Set seaborn style
# sns.set_style("whitegrid")
#
# # Create plot
# sns.lineplot(data=df_target_score, x="cohort", y="avg_risk_rate", ci=95, ax=ax, color="blue", label="Target Score")
#
# for predict_probb in df_predict_score_all_models.filter(regex='avg_score_value_').columns.to_list():
#     sns.lineplot(data=df_predict_score_all_models, x="cohort", y=predict_probb, ci=None, ax=ax,
#                  label=predict_probb)
#
# # Add grid
# ax.grid(True, linestyle="--", axis="y", color="gray", alpha=0.5)
# # plt.axvline(x=split_date, color='black', linestyle='--')
#
# # Iterate over split_end_dates and add vertical lines
# for ii, date in enumerate(split_end_dates):
#     plt.axvline(x=date, color="black", linestyle="--", label=f"Splits_{ii + 1}")
#
# # Set plot title and axis labels
# ax.set_title("Average Risk Rate by Cohort and Score Type")
# ax.set_xlabel("Dates")
# ax.set_ylabel("Average Risk Rate")
# ax.set_ylim(0, 0.25)
#
# # Adjust legend position
# ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#
# # Rotate x-axis labels by 45 degrees
# plt.xticks(rotation=45)
#
# # Add tight layout
# plt.tight_layout()
# plt.savefig(os.path.join("dvc_plots", "temporal_calibration", "temporal_calibration_all_models.png"), dpi=600,
#             bbox_inches="tight")
# # Show plot
# plt.show()
