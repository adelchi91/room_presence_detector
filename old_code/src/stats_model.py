import os
import pathlib
import sys

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import optbinning
import pandas as pd
import sklearn.calibration
import sklearn.inspection
import sklearn.metrics
import tabulate
import yaml
from yc_younipy.metrics.model.calibration import calibration_error
from yc_younipy.metrics.model.roc_auc import gini_computation as gini
from yc_younipy.scorebands import compute_scoreband, compute_scoreband_binning_table

# sys.path.append("./src/")
from helpers_local.helpers_model_2_1 import v2_features  # NOQA
from pipeline_files.helpers import v3_features  # NOQA
from utils import compute_global_figures, is_notebook, pprint_dataframe

"""
stats analyses such as performance, calibration, roc curves and so forth
"""


if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython stats_model.py data/dataframe_for_analytics/X.joblib file")
    sys.exit(1)


sklearn.set_config(transform_output="pandas")

pd.set_option("display.max_rows", 200)
data = pathlib.Path("data")
preprocess = data / "preprocess"
stats_data_dir = data / "stats_model"
stats_data_dir.mkdir(exist_ok=True)

# loading files
payloads_dataset_path = sys.argv[1]
pipeline_path = sys.argv[2]

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

target = params["target"]
test_size = params["validation_size"]
# Dataframes of interest
X = joblib.load(payloads_dataset_path)
X.rename(
    columns={"remainder__contract_reference": "contract_reference", "remainder__application_date": "application_date"},
    inplace=True,
)
y = X[target].astype(float)
pipeline = joblib.load(pipeline_path)
pipeline = pipeline.best_estimator_.named_steps["clf"]

# X0 = joblib.load(preprocess / "gmv" / "X.joblib")
# X = X0.dropna(subset=(target,))
# y = X[target].astype(float)

X_train = X.copy()
X_val = X_train.copy()
y_train = y.copy()
y_val = y_train.astype(float).copy()
print("# Train dataset")
X_preprocessed = (
    pipeline.named_steps["preprocess"]
    .transform(X)
    .rename(
        columns={
            "remainder__contract_reference": "contract_reference",
            "remainder__application_date": "application_date",
        }
    )
)
X_preprocessed[target] = y.copy()
table_summary = compute_global_figures(X_preprocessed, target).hide(axis="index")
pprint_dataframe(table_summary)
table_summary.data.to_csv(stats_data_dir / "table_summary.csv")
models_dir = data / "models"
models_dir.mkdir(exist_ok=True)
clf_v2_1 = joblib.load(models_dir / "v2.1.joblib")
clf_v3_1 = joblib.load(data / "cross_validation" / "v3.1.joblib")
clf_v3_1_full = joblib.load(data / "models" / "v3.1.joblib")

os.makedirs(os.path.join("data", "stats_model"), exist_ok=True)

rows = []
clfs = [("v2.1", clf_v2_1, v2_features), ("v3.1", clf_v3_1, v3_features)]


def compute_performance(clfs, X, y, iso_preacc=False):
    """
    Computes overall performance stats in different scenarios:
    1. At iso preacc, i.e. best 25% of clients
    2. Full
    @param clfs: classifier chosen
    @param X: Covariate dataframe
    @param y: target
    @param iso_preacc: boolean to impose condition iso-preacc
    @return:
    """
    X[target] = y.copy()
    for version, clf, subfeatures in clfs:
        X["predict_prob"] = clf.predict_proba(X)[:, 1]
        # performance at iso_preacc
        if iso_preacc:
            # Calculate the threshold to select the top 25% best clients
            threshold = np.percentile(X["predict_prob"], 30)
            # Select the data points with predicted probabilities less than or equal to the threshold
            df_filtered = X[X["predict_prob"] <= threshold]
            y_pred = df_filtered["predict_prob"].copy()
            y = df_filtered[target].copy()
        else:
            y_pred = clf.predict_proba(X)[:, 1]
        rows.append(
            {
                "version": version,
                "gini": gini(y, y_pred),
                "brier": sklearn.metrics.brier_score_loss(y, y_pred),
                "logloss": sklearn.metrics.log_loss(y, y_pred),
                "calib_error": calibration_error(y, y_pred),
                "count": y.count(),
                "defaults": y.sum(),
                "default_rate": y.mean(),
            }
        )
    df = (
        pd.DataFrame(rows)
        .style.format(
            {
                "default_rate": "{:.2%}",
                "brier": "{:.4}",
                "defaults": "{:0}",
            },
            precision=3,
            thousands=",",
        )
        .hide(axis="index")
    )
    print("# Performance")
    pprint_dataframe(df, headers="keys", tablefmt="psql", floatfmt=".3f")
    df.data.to_csv(os.path.join("data", "stats_model", f"df_stats_iso_preacc_{iso_preacc}.csv"))
    print()
    return df


def compute_roc_curves(clfs, X, y, folder, title=None):
    """
    Roc curve
    @param clfs: Classifier pipeline
    @param X: Covariate dataframe
    @param y: target vector column
    @param folder: folder where the graph is saved
    @param title: title chosen
    @return:
    """
    fig, ax = plt.subplots()
    for version, clf, subfeatures in clfs:
        y_pred = clf.predict_proba(X)[:, 1]
        fpr, tpr, _ = sklearn.metrics.roc_curve(y, y_pred)
        gini = 2 * sklearn.metrics.roc_auc_score(y, y_pred) - 1
        plt.plot(fpr, tpr, label=f"{version}, Gini={gini:.3f}")

    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{folder}/roc_curve.png")


def compute_scorecard(clf, X, y):
    """
    Optimal binning score card showint all of the different modalities chosen by OptimalBinning and their relevant
    stats, e.g. IV, WOE...
    @param clf: Classifier pipeline
    @param X: Covariate dataframe
    @param y: target vector column
    @return:
    """
    logistic = clf.named_steps["logistic"]  # clf[-1]
    encoder = clf.named_steps["encoder"]  # clf[-2]
    if isinstance(encoder, optbinning.BinningProcess) and isinstance(logistic, sklearn.linear_model.LogisticRegression):
        scorecard = optbinning.Scorecard(binning_process=encoder, estimator=logistic)
        scorecard.fit(X, y, metric_missing="empirical")
        df = (
            scorecard.table("detailed")
            .reset_index(drop=True)
            .style.format(
                {"Event rate": "{:.2%}", "Count (%)": "{:.2%}"},
                thousands=",",
                precision=4,
            )
            .background_gradient(subset="Points", cmap="RdYlGn_r")
            .hide(axis="index")
        )
    else:
        dfs = []
        for variable, coef in zip(logistic.feature_names_in_, logistic.coef_[0]):
            dfs.append(
                encoder.get_binned_variable(variable)
                .binning_table.build()
                .drop("Totals")
                .query("Count > 0")[["Bin", "WoE"]]
                .assign(variable=variable, coef=coef)
            )
        df = (
            pd.concat(dfs)
            .assign(coef=lambda X: X["coef"] * X["WoE"])[["variable", "Bin", "coef"]]
            .rename(columns={"Bin": "bin"})
            .reset_index(drop=True)
            .style.background_gradient(subset="coef", cmap="RdYlGn_r")
            .hide(axis="index")
        )
    print("# Scorecard")
    pprint_dataframe(df, headers="keys", tablefmt="psql", floatfmt=".3f")
    # save here file
    df.data.to_csv(stats_data_dir / "optbinning_scorecard.csv")
    print()


def compute_calibration_curves(clfs, X, y, folder):
    fig, ax = plt.subplots()
    pmax = 0.4
    ax.set_xlim((0, pmax))
    ax.set_ylim((0, pmax))
    for version, clf, subfeatures in clfs:
        y_pred = clf.predict_proba(X)[:, 1]
        prob_true, prob_pred = sklearn.calibration.calibration_curve(y, y_pred, strategy="quantile", n_bins=5)
        plt.plot(prob_pred, prob_true, label=f"{version}")
    plt.plot(
        np.arange(0, pmax, 0.01),
        np.arange(0, pmax, 0.01),
        "--k",
        label="Perfect calibration",
    )
    plt.legend()
    plt.xlabel("Prediction")
    plt.ylabel("Observation")
    plt.tight_layout()
    plt.savefig(f"{folder}/calibration_curve.png")


def compute_calibration_temporal(clfs, X, folder, validation_date=None):
    pmax = 0.4
    ax = (
        pd.DataFrame(
            {
                "application_date": X["application_date"],
                "dn2_4": X["dn2_4"],
                "dn2_6": X["dn2_6"],
                "dn3_12": X["dn3_12"],
                **{version: clf.predict_proba(X)[:, 1] for version, clf, subfeatures in clfs},
            }
        )
        .groupby(pd.Grouper(key="application_date", freq="1Q"))
        .mean()
        .plot(ylim=(0, pmax), figsize=(12, 4))
    )
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    if validation_date:
        ax.vlines(validation_date, 0, 1)
        ax.text(validation_date, pmax * 0.9, "Validation")
    plt.tight_layout()
    plt.savefig(f"{folder}/calibration_trend.png")
    if is_notebook():
        plt.show()


def compute_importances(clfs, X, y):
    print("# Feature importance")
    print()
    for version, clf, subfeatures in clfs:
        feature_importance = sklearn.inspection.permutation_importance(
            clf,
            X[subfeatures],
            y,
            random_state=0,
            scoring=sklearn.metrics.make_scorer(gini, needs_proba=True),
        )
        feature_importance = (
            pd.DataFrame(
                {
                    "importances_mean": feature_importance["importances_mean"],
                    "importances_std": feature_importance["importances_std"],
                },
                index=subfeatures,
            )
            .sort_values("importances_mean", ascending=False)
            .rename_axis("variable")
            .head(20)
        )
        feature_importance["importances_cum"] = (
            feature_importance["importances_mean"].cumsum() / feature_importance["importances_mean"].sum()
        )
        print(f"## {version}")
        print(tabulate.tabulate(feature_importance, headers="keys", tablefmt="psql", floatfmt=".4f"))
        print()


def compute_scorebands(clfs, X, y, binning_params={"max_pvalue": 0.1}):
    print("# Score bands")
    print()
    for version, clf, subfeatures in clfs:
        print(f"## {version}")
        optb, table = compute_scoreband_binning_table(
            clf, X, y, recalibrate=True, binning_params=binning_params, use_proba=True
        )
        bands = table.query("band != 'Run-off'")["band"].unique()
        coverage = table.query("band != 'Run-off'")["Count (%)"].sum()
        a1_a4 = table.query("'A1' <= band <= 'A4'")["Count (%)"].sum()
        a1_a6 = table.query("'A1' <= band <= 'A6'")["Count (%)"].sum()
        a1_a8 = table.query("'A1' <= band <= 'A8'")["Count (%)"].sum()
        print(
            f"Found {len(bands)} unique bands: {bands}, coverage={coverage:.0%}, [A1; A4]={a1_a4:.0%}, [A1; A6]={a1_a6:.0%}, [A1; A8]={a1_a8:.0%}"
        )
        pprint_dataframe(
            table[
                [
                    "Bin",
                    "Count",
                    "Count (%)",
                    "Event",
                    "Event rate",
                    "prediction",
                    "band",
                ]
            ].style.hide(axis="index"),
            headers="keys",
            tablefmt="psql",
            floatfmt=".4f",
        )
        X[version] = compute_scoreband(clf.predict_proba(X)[:, 1], optb, table).values
        df = (
            pd.DataFrame(
                {
                    "bins": optb.transform(clf.predict_proba(X_val)[:, 1], metric="bins", show_digits=4),
                    "pred": clf.predict_proba(X_val)[:, 1],
                    "true": y_val,
                }
            )
            .merge(table[["Bin", "band"]], left_on="bins", right_on="Bin")
            .assign(
                band=lambda df: df["band"].apply(
                    lambda band: "A1-A6" if band in ("A1", "A2", "A3", "A4", "A5", "A6") else "A7+"
                )
            )
            .groupby("band")
            .agg(["sum", "count", "mean"])
        )
        pprint_dataframe(df, headers="keys", tablefmt="psql", floatfmt=".4f")


dvc_plots = pathlib.Path("dvc_plots")
stats_dir = dvc_plots / "stats"
stats_dir.mkdir(exist_ok=True)
validation_dir = stats_dir / "validation"
validation_dir.mkdir(exist_ok=True)

binning_params = {"max_pvalue": 0.1}
compute_scorecard(clf_v3_1_full, X_preprocessed, y)
compute_performance(clfs, X_train, y, iso_preacc=False)
compute_performance(clfs, X_train, y, iso_preacc=True)
# compute_importances(clfs, X_val, y_val)


# compute_roc_curves(clfs, X_train, y_train, validation_dir, "ROC Curve on train")
# compute_roc_curves(clfs, X_val, y_val, validation_dir, "ROC Curve on validation")

compute_calibration_curves(clfs, X, y, validation_dir)
compute_calibration_temporal(clfs, X, validation_dir, X_val["application_date"].min())

compute_scorebands(clfs, X_train, y_train, binning_params)

# all_dir = stats_dir / "all"
# all_dir.mkdir(exist_ok=True)
# compute_calibration(clfs, X, y, all_dir)
# compute_roc_curves(clfs, X, y, all_dir, "All")


clf = clfs[-1][1]
clf.fit(X, y)
compute_scorebands(clfs, X, y, binning_params)
