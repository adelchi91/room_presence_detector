import os
import sys

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pandas import DataFrame, concat, options
from sklearn import metrics
from yc_younipy.metrics.model.calibration import plot_calibration_curve
from yc_younipy.metrics.model.logistic_regression import lr_summary
from yc_younipy.metrics.model.roc_auc import plot_roc_curve

from pipeline_files.helpers import v3_cat_features, v3_features  # NOQA

"""
stats analyses such as performance, calibration, roc curves and so forth
"""

options.display.max_columns = 40
options.display.width = 1000

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython stats_model.py training_dataset_path validation_dataset_path pipeline_path variables_file"
    )
    sys.exit(1)

# Load files
training_dataset_path = sys.argv[1]
pipeline_path = sys.argv[2]

# Load files
df_train = joblib.load(open(training_dataset_path, "rb"))
pipeline = joblib.load(open(pipeline_path, "rb"))
pipeline = pipeline.best_estimator_.named_steps["clf"]

# Load variables set
form_variables = v3_features
numerical_vars = list(set(v3_features) - set(v3_cat_features))
cat_feature = v3_cat_features

# global parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
target = params["target"]

# Save outputs
os.makedirs(os.path.join("dvc_plots", "stats_model2"), exist_ok=True)

###################
# Local functions #
###################


def get_scope_data(df, target):
    """
    select the dataframe rows where target does not have any missing values
    :param df: dataframe we are working with
    :param target: target value we want to predict during our model creation
    :return: dataframe properly filtered
    """
    if target is None:
        raise ValueError("target parameter must not be None")
    return df[~df[target].isna()].copy()


def metrics_computation(df, indicator, prediction_column):
    """
    Computes the auc value for the given dataframe as defined by the dataframe_id (e.g. validation or train_test).

    :param df: Input dataframe needed to train the model.
    :param indicator: column to apply auc on
    :return: auc
    """
    y = df[indicator]
    # predicting model probabilities
    y_prob = df[prediction_column]
    # Computing AUC
    fpr, tpr, thresholds = metrics.roc_curve(y, y_prob)

    res_auc = metrics.auc(fpr, tpr)
    res_brier = metrics.brier_score_loss(y, y_prob)
    return res_auc, res_brier


def model_evaluation(df, target, split_col):
    """
    Stores computed metrics values (e.g. auc) for each dataframe id, as well as the target percentage, the number of
    rows and the number of columns in a log dataframe df_log.

    :param df: Input dataframe needed to train the model.
    :return: df_log
    """
    df_tmp = df.copy()
    df_log = DataFrame()
    # test_indicators can be a list for all of the indicators on which we want to test the model
    test_indicators = [target]
    for indicator in list(set(test_indicators)):
        results_auc = {}
        results_brier = {}
        df_filtered = get_scope_data(df_tmp.copy(), indicator)
        for sub_df in df_filtered[split_col].unique().tolist():
            results_auc[sub_df], results_brier[sub_df] = metrics_computation(
                df_filtered[df_filtered[split_col] == sub_df], indicator, "predict_prob"
            )

        df_log_indicator = (
            df_filtered.groupby(split_col, as_index=False)
            .agg(n_contract=(indicator, "count"), n_target=(indicator, "sum"))
            .loc[lambda x: x[split_col] != "target_not_defined"]
            .assign(
                target_rate=lambda x: x["n_target"] / x["n_contract"],
                auc=lambda x: x[split_col].apply(lambda y: results_auc[y]),
                brier=lambda x: x[split_col].apply(lambda y: results_brier[y]),
                gini=lambda x: 2 * x["auc"] - 1,
                indicator=indicator,
            )
        )
        df_log = concat([df_log, df_log_indicator])

    return df_log


def score_card_lr(pipeline, X, model):
    """
    scorecard for the logistic regression model
    @param pipeline: full pipeline including preprocessing steps and model
    @param X: covariate dataframe
    @param model: classifier model
    @return: score_card and intercept as two dataframes
    """
    feature_names = model.feature_names_in_
    # reproducing preprocessing steps to feed correct dataframe to model
    X_preprocessed = pipeline.named_steps["preprocess"].transform(X)
    X_preprocessed = pipeline.named_steps["encoder"].transform(X_preprocessed[feature_names])
    X_preprocessed = pipeline.named_steps["scaler"].transform(X_preprocessed)
    # computing scorecard for logistic regression with model object and preprocessed dataframe
    df_score_card, intercept = lr_summary(model, X_preprocessed)
    # sorting scorecard based on coefficients values
    df_score_card = df_score_card.sort_values(by=["coefficients"], ascending=False)
    # recording intercept in a separate dataframe
    df = DataFrame()
    df["intercept"] = intercept
    return df_score_card, df


def correlation_matrix_plot(df):
    # Compute the correlation matrix
    corr_matrix = df.corr()

    # Set up the matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Generate a heatmap of the correlation matrix
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".1f",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        ax=ax,
        annot_kws={"fontsize": 8},
    )

    # Rotate the x-axis tick labels by 30 degrees and decrease font size
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    # Decrease font size of y-axis tick labels
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    # Set the axis labels and title
    ax.set_title("Correlation Matrix Plot")
    ax.set_xlabel("Features")
    ax.set_ylabel("Features")
    # Adjust the spacing between subplots
    plt.tight_layout()
    # Save the plot as a file
    print("hi")
    fig.savefig("dvc_plots/stats_model2/correlation_matrix.png", dpi=600, bbox_inches="tight", pad_inches=0.2)
    # Show the plot
    plt.show()
    return


# subset where target is defined
df_train = df_train.dropna(subset=(target,)).copy()
# predict proba
train_predictions = pipeline.predict_proba(df_train.drop(columns=[target]))[:, 1]

# Save outputs
os.makedirs(os.path.join("dvc_plots", "stats_model2"), exist_ok=True)

# # SHAP values plot
# with open(os.path.join("dvc_plots", "shap_summary_plot.png"), 'wb') as f:
#     df_val_preprocessed = df_val.sample(n=20) #DataFrame(pipeline['preprocessing'].transform(df_val[form_variables]),
#         #                             columns=pipeline['preprocessing'].get_feature_names_out(form_variables+cat_feature)).sample(
#         # n=20)
#     columns = [form_variables+cat_feature]
#     dtype = df_val_preprocessed.dtypes[form_variables+cat_feature].apply(lambda x: x.name).to_dict()
#     pred_fun = lambda x: pipeline.predict_proba(DataFrame(x, columns=columns).astype(dtype=dtype))[:, 1]
#     f.write(shap_summary_plot(pred_fun, X=df_val_preprocessed[form_variables+cat_feature], explainer_type="function"))

# ROC curve plot
roc_curve = plot_roc_curve(list(df_train[target].values), list(train_predictions), (["train"] * len(train_predictions)))
with open(os.path.join("dvc_plots", "stats_model2", "roc_curve.png"), "wb") as f:
    f.write(roc_curve)


# Performance summary
df_performance_summary = DataFrame()
df_train = df_train.assign(dataframe="train_test", predict_prob=train_predictions)
df_full = df_train.copy()
# df_train[target] is here of type int64, which causes problems to scikit learn. I have to convert it in an int
df_full[target] = df_full[target].astype("int")
df_full[target] = df_full[target].astype("int")
df_performance_summary = model_evaluation(df_full, target, "dataframe")
df_lr_summary, df_intercept = score_card_lr(pipeline, df_full, model=pipeline["logistic"])

# Calibration curves plot
df_train[target] = df_train[target].astype("int")

calibration_curve_train = plot_calibration_curve(
    df_train, observed_target=target, predicted_target="predict_prob", strategy="quantile", n_bins=10
)

with open(os.path.join("dvc_plots", "stats_model2", "calibration_curve_train.png"), "wb") as f:
    f.write(calibration_curve_train)


# correlation matrix
os.makedirs(
    os.path.join(
        "dvc_plots",
        "stats_model2",
    ),
    exist_ok=True,
)
# preprocessing data so that correlation matrix can be created
X_preprocessed = pipeline.named_steps["preprocess"].transform(df_full)
correlation_matrix_plot(X_preprocessed[pipeline["logistic"].feature_names_in_])

# Save outputs
os.makedirs(os.path.join("data", "stats_model2"), exist_ok=True)
os.makedirs(os.path.join("data", "stats_model2"), exist_ok=True)
df_performance_summary.to_csv("data/stats_model2/performance_summary.csv")
df_lr_summary.to_csv("data/stats_model2/score_card.csv")
df_intercept.to_csv("data/stats_model2/intercept.csv")
