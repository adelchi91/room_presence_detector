import os
import pathlib
import sys
from datetime import timedelta
from io import BytesIO
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import optbinning
import pandas as pd
import plotnine
import seaborn as sns
import sklearn.compose
from mizani.utils import get_timezone
from numpy import array
from pandas import DataFrame, Series, to_datetime
from pandas.api.types import is_numeric_dtype
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from yc_younipy.plot import yc_colors
from yc_younipy.preprocessing.column_transformer_df import ColumnTransformerDF
from yc_younipy.preprocessing.most_frequent_binning import MostFrequentBinning
from yc_younipy.preprocessing.quantile_binning import QuantileBinning

from utils import compute_global_figures, pprint_dataframe

# CONSTANTS
AGE_ARBITRARY_BUCKETS = [(30, "[18; 30["), (45, "[30; 45["), (55, "[45; 55["), (65, "[55; 65[")]
CHOSEN_TARGET = "dn3_12"
CALIBRATION_VARIABLES = [
    ("simpleimputer-1__bank_age", "optbinning", {"monotonic_trend": "auto_asc_desc"}),
    ("remainder__bank_code", "optbinning", {"dtype": "categorical"}),
    ("remainder__marital_status_code", "optbinning", {"dtype": "categorical"}),
    ("remainder__business_provider_code", "most_frequent", {"top_n": 4, "output_labels": True}),
    ("remainder__housing_code", None, None),
    ("remainder__main_net_monthly_income", "optbinning", {"monotonic_trend": "auto_asc_desc"}),
    ("remainder__marital_status_code", None, None),
    ("simpleimputer-2__mortgage_amount", "optbinning", {"monotonic_trend": "auto_asc_desc"}),
    ("simpleimputer-1__personal_age", None, None),
]
CPM_GRAPHS_BINS = 5


def compute_age_wrapper(date_from, date_to):
    """
    method used to compute an age difference in years.
    @param date_from: intial date
    @param date_to:  ending date
    @return: age in years
    """
    try:
        dates = [
            pd.to_datetime(date, utc=True, errors="coerce")
            if not isinstance(date, pd.Timestamp)
            else date.dt.tz_localize("utc")
            for date in [date_from, date_to]
        ]
        return (dates[1] - dates[0]) / timedelta(days=365)
    except ValueError:
        res = np.nan
    return res


def out_of_time_split(df, target, test_size):
    """
    Out of time split using test size instead of threshold date
    @param df: input dataframe: should contains feature columns + target
    @param target: target column, usually a risk indicator e.g. dn3_12
    @param test_size: test size, i.e. a percentage number going from 0 to 1.
    @return: covariate dataframe X, target dataframe y and their correspondent training and validation datasets.
    """
    X = df.dropna(subset=(target,))
    y = X[target].astype(float)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)
    print("# Train dataset")
    pprint_dataframe(compute_global_figures(X_train, target).hide(axis="index"))
    print("# Validation dataset")
    pprint_dataframe(compute_global_figures(X_val, target).hide(axis="index"))
    return X, y, X_train, X_val, y_train, y_val


def plot_cpm_graphs(X, target_data, new_version, model_chosen, apply_filter, apply_decile):
    """
    We are producing plots for CALIBRATION_VARIABLES, which are variables that CPM and investors tend to look at.
    We want to make sure that the risk of such variables is well-captured by the score. If it is the case, then
    all the different modalities/buckets present for each variable, correlates well with the score. Ideally, curves
    produced ought to be as close as possible.
    @param X: Input dataframe containing model predictions present in "predict_prob" column.
    @param target_data: target column (i.e. a vector). Usually a default such as dn3_12
    @param new_version: the name of the candidate model (e.g. V2), which is shown underneath the x-axis for each graph
    @param model_chosen: Redundant varaible, similar to new_version, but with a longer name, used to name the output
    file
    @param apply_filter: string value used in the naming of graphs produced to inform if a particular filter was applied
    on covariate dataset X.
    @param apply_decile: string value used in the naming of graphs produced to inform if only the first n-deciles of the
    score were considered. See cpm_graphs.py stage for more details.
    @return:
    """

    def binarize_age(age):
        age_ranges = AGE_ARBITRARY_BUCKETS
        for cutoff, label in age_ranges:
            if age < cutoff:
                return label
        return "[65+]"

    def data_formatting(df):
        if "personal_age" in df.columns:
            df["personal_age"] = df["personal_age"].apply(binarize_age)
            df["personal_age"] = df["personal_age"].astype("string")
        if "mortgage_amount" in df.columns:
            df["mortgage_amount"] = df["mortgage_amount"].round(0)
        if "business_provider_code" in df.columns:
            df["business_provider_code"] = df["business_provider_code"].astype("string")
        return df

    # definition of output folders
    sklearn.set_config(transform_output="pandas")
    dvc_plots_directory = pathlib.Path("dvc_plots") / "score_dependencies"
    dvc_plots_directory.mkdir(exist_ok=True)
    # fixed parameters
    target = CHOSEN_TARGET
    bins = CPM_GRAPHS_BINS
    versions = (new_version,)
    variables = CALIBRATION_VARIABLES  # variables of interest for CPM
    # KBinsDiscretizer is a sklearn transformer that bins continuous features into discrete bins.
    discretizer = KBinsDiscretizer(bins, encode="ordinal")
    # defining buckets for the predicted score
    X[new_version] = X.predict_prob.copy()
    X[new_version] = discretizer.fit_transform(X[[new_version]])[new_version]
    # definition of dtypes and bucketization of age
    X = data_formatting(X)
    # Graphs production in two steps: 1. buckets creation according to binning_type as defined in CALIBRATION_VARIABLES
    # and 2. seaborn pointplot graph creation.
    for variable, binning_type, kwargs in variables:
        print(f"Plotting score dependency for variable {variable}")
        index = X.columns.get_loc(variable)
        if binning_type == "quantile":
            X[f"{variable}_bin"] = QuantileBinning(**kwargs).fit_transform(X.iloc[:, index])[:, 0]
        elif binning_type == "most_frequent":
            X[f"{variable}_bin"] = MostFrequentBinning(**kwargs).fit_transform(X[[variable]])
        elif binning_type == "optbinning":
            X[f"{variable}_bin"] = optbinning.OptimalBinning(**kwargs).fit_transform(
                X.iloc[:, index], target_data.astype(float), metric="bins"
            )
        else:
            X[f"{variable}_bin"] = X.iloc[:, index]
        fig, axes = plt.subplots(1, len(versions), figsize=(20, 5))
        for v, index in enumerate(versions):
            sns.pointplot(data=X, x=index, y=target, hue=f"{variable}_bin", n_boot=1000, dodge=True)
            plt.axhline(y=0.10, linestyle="--", color="k")
            plt.tight_layout()
            os.makedirs(os.path.join("dvc_plots", "cpm_graphs"), exist_ok=True)
            plt.savefig(
                os.path.join(
                    "dvc_plots",
                    "cpm_graphs",
                    f"score_dependency{model_chosen}_{variable}_filters_{apply_filter}_{apply_decile}.png",
                ),
                dpi=600,
                bbox_inches="tight",
            )
    return


def binning(alt_w_pred_converted, numerical_vars, categorical_vars):
    """
    Binning method that can be applied on a dataframe containing both numerical and categorical variables. Method is
    used to preprocess data given to univariate and temporal graphs for instance.
    @param alt_w_pred_converted:
    @param numerical_vars: list of numerical variables
    @param categorical_vars: list of categorical variables
    @return:
    """
    # Define transformers for numerical and string columns
    num_transformer = Pipeline(steps=[("binning", QuantileBinning(nb_bins=4, output_labels=True))])
    str_transformer = Pipeline(
        steps=[
            ("imputer", MostFrequentBinning(top_n=4, output_labels=True)),
        ]
    )
    # Define preprocessor to apply transformers to respective columns
    preprocessor = ColumnTransformerDF(
        [
            ("num", num_transformer, numerical_vars),
            ("str", str_transformer, categorical_vars),
        ],
        remainder="passthrough",
    )
    # Transform data using pipeline
    output = preprocessor.fit_transform(alt_w_pred_converted)
    return output


def get_features_out(ft, input_features):
    """
    Function returning output feature names when passed input feature names
    @param ft: dataframe
    @param input_features: feature names
    @return:
    """
    return list(input_features)


def create_ensembles(df, split_type, validation_date, validation_size):
    """
    method to carry out either an out of time split, with a threshold date, i.e. validation_date, or a random split
    using the validation size as test_size.
    @param df: Input dataframe on which we apply the split
    @param split_type: can be either "out_of_time", "train_test" or "no_split"
    @param validation_date: threshold date
    @param validation_size: percentage of corresponding to the validation dataset. Must be a number between 0 to 1.
    @return:df_train, df_val, i.e. training and validation datasets
    """
    if split_type == "out_of_time":
        print("Time threshold chosen : ", validation_date)
        df = df.assign(split=lambda x: np.where(x["application_date"] < validation_date, "train_test", "validation"))
        df_train = df.loc[lambda x: x["split"] == "train_test"].sort_values("application_date")
        df_val = df.loc[lambda x: x["split"] == "validation"].sort_values("application_date")

    elif split_type == "train_test":
        df_train, df_val, _, _ = train_test_split(df, [0] * df.shape[0], test_size=validation_size, random_state=42)

    elif split_type == "no_split":
        df_train = df
        df_val = None

    else:
        print("ERROR: this split method isn't implemented yet !")
        sys.exit(1)

    return df_train, df_val


def handle_numerical_missing(df):
    """
    Creates new columns following the pattern <feature>_IS_NULL with binary values.
    That way, we can replace the missing values within numerical features without losing this information
    @param df: Input dataframe
    @return: preprocessed dataframe
    """
    new_df = df.copy()
    for num_col in [col for col in df.columns]:
        new_df[f"{num_col}_IS_NULL"] = pd.isnull(df[num_col]).astype(int)

    output_features = df.columns.tolist() + [col + "_IS_NULL" for col in df.columns.values]
    return new_df[output_features]


def get_numerical_features_after_missing_treatment(tf, input_features):
    """
    Get the feature names after handle_numerical_missing was launched

    @param tf: Input dataframe
    @param input_features: Features list
    @return: list of features
    """
    return list(input_features) + [feature + "_IS_NULL" for feature in input_features]


def calibration_by_feature(
    df_: pd.DataFrame,
    feature_name: str,
    prediction_name: str,
    event_name: str,
    pct: float = 0.9,
    n: int = 5,
    bonferonni_correction: bool = True,
) -> pd.DataFrame:
    """
    Print the calibration test of a prediction for a feature with a confidence interval.
    :param event_name: name of the event, e.g. dn3_12
    :param prediction_name: name of the prediction
    :param df_: dataframe with the feature 'feature_name', the score 'prediction_name' and the event 'event_name'
    :param feature_name: feature used to split the population
    :param pct: optional confidence interval, default value equals 0.90
    :param bonferonni_correction: optional to correct the probability in function of the number of tests performed
    :param n: max number of splits, default value equals 5
    :return: dataframe with test results of each modality of 'feature_name'
    """
    d = df_[[feature_name, prediction_name, event_name]].copy()

    if is_numeric_dtype(d[feature_name]) and len(d[feature_name].unique()) > n:
        d[feature_name] = pd.qcut(d[feature_name], n, duplicates="drop").astype(str)

    n_test = min(n, len(d[feature_name].unique()))
    pct = 1 - (1 - pct) / n_test if bonferonni_correction else pct

    d = (
        d.groupby(feature_name)
        .agg(n_obs=(prediction_name, "count"), avg_pred=(prediction_name, "mean"), avg_dr=(event_name, "mean"))
        .sort_values(by="n_obs", ascending=False)
        .head(n)
        .reset_index()
    )

    d = d[d["n_obs"] > 0]

    d["a"] = d["avg_pred"] * d["n_obs"]
    d["b"] = (1 - d["avg_pred"]) * d["n_obs"]
    d["lower_bound"] = d.apply(lambda x: beta.ppf((1 - pct) / 2, x.a, x.b + 1), axis=1)
    d["upper_bound"] = d.apply(lambda x: beta.ppf((1 + pct) / 2, x.a + 1, x.b), axis=1)
    d["success"] = (d["lower_bound"] <= d["avg_dr"]) & (d["avg_dr"] <= d["upper_bound"])

    d.drop(columns={"a", "b"}, inplace=True)
    d.insert(0, "feature_name", feature_name)
    d.rename(columns={feature_name: "value"}, inplace=True)
    d["value"] = d["value"].astype(str)

    d["avg_pred"] = (d["avg_pred"] * 100).map("{:.2f}".format) + "%"
    d["avg_dr"] = (d["avg_dr"] * 100).map("{:.2f}".format) + "%"
    d["lower_bound"] = (d["lower_bound"] * 100).map("{:.2f}".format) + "%"
    d["upper_bound"] = (d["upper_bound"] * 100).map("{:.2f}".format) + "%"

    return d.set_index(["feature_name", "value"])


def _create_pop_over_time_data(
    x: Union[Series, array], x_date: Union[Series, array], date_cohort_type: str = "D"
) -> DataFrame:
    """
    Create pop over time data
    :param x: Observations to plot
    :param x_date: Corresponding date values
    :param date_cohort_type: Type of period to adopt during plotting. Example "D", "5D", "W", "M"...
    :return: Infos over time regarding the studied variable (number of observations, proportion of observations)
    """

    df = DataFrame({"variable": array(x), "date": array(x_date)})

    df["date"] = to_datetime(df["date"]).dt.to_period(date_cohort_type).dt.to_timestamp()

    df_agg = (
        df.assign(n_contract_all=lambda x: x.groupby(["date"])["variable"].transform("count"))
        .groupby(["date", "variable"], as_index=False, observed=True)
        .agg(n_contract=("variable", "count"), n_contract_all=("n_contract_all", "first"))
        .assign(prop_contrat=lambda x: x["n_contract"] / x["n_contract_all"])
    )

    variable_values = DataFrame({"variable": df_agg["variable"].unique()})
    date_values = DataFrame({"date": df_agg["date"].unique()})

    return (
        date_values.join(variable_values, how="cross")
        .merge(df_agg, left_on=["date", "variable"], right_on=["date", "variable"], how="left")
        .fillna(0)
    )


def plot_pop_over_time(
    x: Union[Series, array],
    x_date: Union[Series, array],
    date_cohort_type: str,
    variable_name: str,
    use_prop: bool = False,
) -> bytes:
    """
    Create pop over time data
    :param x: Observations to plot.
    :param x_date: Corresponding date values.
    :param date_cohort_type: Type of period to adopt during plotting. Example "D", "5D", "W", "M"...
    :param variable_name: Name of the plotted variable.
    :return: Bytes plot
    """
    df = _create_pop_over_time_data(x, x_date, date_cohort_type)

    if use_prop:
        y = "prop_contrat"
    else:
        y = "n_contract"
    n_modalities = len(df["variable"].unique())
    df["variable"] = df["variable"].astype(str)
    if n_modalities > 25:
        print("Categorical variable has more than 25 modalities ! This plot has been canceled.")
        return None
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(get_timezone(df["date"]))
        plot = (
            plotnine.ggplot(df, plotnine.aes(x="date", y=y, fill="variable"))
            + plotnine.geom_area(position="stack")
            + plotnine.scale_fill_manual(values=yc_colors(range(n_modalities)))
            + plotnine.theme(
                legend_title=plotnine.element_blank(),
                axis_title_y=plotnine.element_blank(),
                legend_position="right",
                axis_text_x=plotnine.element_text(angle=20, hjust=1),
            )
            + plotnine.labs(title=f"{variable_name} w.r.t time")
        )

        figfile = BytesIO()
        plt.pause(1e-13)  # hack to prevent Tinker exceptions poping ...
        plotnine.ggsave(plot, figfile, format="png", units="cm", height=14, width=24)
        bytes_plot = figfile.getvalue()
        plt.close("all")
        return bytes_plot
