import os

import matplotlib.pyplot as plt
import numpy as np
import optbinning
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

__all__ = ("get_target_dn3_12", "get_rating_dn3_12", "compute_scoreband_binning_table")


def get_target_dn3_12(band):
    return {
        "A1": 0.005,
        "A2": 0.015,
        "A3": 0.025,
        "A4": 0.035,
        "A5": 0.045,
        "A6": 0.058,
        "A7": 0.075,
        "A8": 0.091,
    }.get(band)


def get_rating_dn3_12(proba, use_proba=True):
    """
    Return the rating given the predicted dr3_12.

    :param float proba: Predicted probability.
    """
    if use_proba:
        if proba < 0 or proba > 1:
            return None
        if proba <= 0.01:
            return "A1"
        elif proba <= 0.02:
            return "A2"
        elif proba <= 0.03:
            return "A3"
        elif proba <= 0.04:
            return "A4"
        elif proba <= 0.05:
            return "A5"
        elif proba <= 0.0667:
            return "A6"
        elif proba <= 0.0833:
            return "A7"
        elif proba <= 0.1:
            return "A8"
    else:
        if proba < 0 or proba > 10000:
            return None
        if proba >= 9900:
            return "A1"
        elif proba >= 9800:
            return "A2"
        elif proba >= 9700:
            return "A3"
        elif proba >= 9600:
            return "A4"
        elif proba >= 9500:
            return "A5"
        elif proba >= 9333:
            return "A6"
        elif proba >= 9166:
            return "A7"
        elif proba >= 9000:
            return "A8"
    return "Run-off"


def compute_scoreband_binning_table(clf, X, y, default="dn3_12", recalibrate=False, binning_params={}, use_proba=True):
    """
    Compute scoreband binning table which is optimal binning table enriched with score band information.

    :param clf: Fitted classifier.
    :param X: Training vector.
    :param y: Target relative to X.
    :param dict binning_params: Parameters to provide to optimal binning.
    :param str default: Definition of default.
    :param bool recalibrate: Indicates if recalibration should be applied by using mean default instead of prediction.
    :param bool use_proba: Indicates if proba should be used instead of score.
    """
    optb = optbinning.OptimalBinning(**binning_params)
    y_pred = clf.predict_proba(X)[:, 1]
    if not use_proba:
        y_pred = 10_000 * (1 - y_pred)
    show_digits = 4 if use_proba else 0
    bins = optb.fit_transform(y_pred, y, metric="bins", show_digits=show_digits)
    binning_table = optb.binning_table.build(show_digits=show_digits).query("Count > 0")
    binning_table = binning_table.merge(
        pd.DataFrame({"prediction": y_pred, "bins": bins}).groupby("bins").mean(),
        left_on="Bin",
        right_index=True,
        how="left",
    )
    get_rating = {"dn3_12": get_rating_dn3_12}[default]
    binning_table["band"] = binning_table["Event rate" if recalibrate else "prediction"]
    if recalibrate and not use_proba:
        binning_table["band"] = 10_000 * (1 - binning_table["band"])
    binning_table["band"] = binning_table["band"].apply(lambda pred: get_rating(pred, use_proba))
    return optb, binning_table


def compute_scoreband(x, optb, binning_table, use_proba=True):
    """
    Compute scoreband based on a binning table

    :param x: Prediction vector (proba or score).
    :param OptimalBinning optb: OptimalBinning object returend by `compute_scoreband_binning_table`.
    :param pd.DataFrame binning_table: Table returned by `compute_scoreband_binning_table`.
    :param bool use_proba: Indicates if proba should be used instead of score.
    """
    show_digits = 4 if use_proba else 0
    return binning_table.set_index("Bin").loc[optb.transform(x, metric="bins", show_digits=show_digits), "band"]


def compute_scoreband_for_replay(df):
    df["pd_score"] = 1 - df.score_replay / 10000

    x_bins = df["score_replay"].astype(int)
    y_target = df["dn3_12"]
    optb = optbinning.OptimalBinning(name="score_replay", dtype="numerical", solver="cp", split_digits=0, max_n_bins=20)
    optb.fit(x_bins, y_target)

    df["optimal_scoreband"] = optb.transform(df["score_replay"].values, metric="bins")

    df["avg_score"] = df.groupby("optimal_scoreband")["pd_score"].transform(np.mean)
    df["recommended_category"] = df["pd_score"].apply(lambda x: get_rating_dn3_12(x))

    df_results = df.groupby("optimal_scoreband").agg(
        pd_score=("pd_score", np.mean),
        min_score=("score_replay", np.min),
        max_score=("score_replay", np.max),
        avg_score=("score_replay", np.mean),
        var_score=("score_replay", np.var),
        n_obs=("contract_reference", "count"),
        event=("dn3_12", sum),
    )

    df_results["dn3_12"] = df_results.event / df_results.n_obs

    df_results["recommended_category"] = df_results["pd_score"].apply(lambda x: get_rating_dn3_12(x))
    return df, df_results


# %%
def plot_confusion_matrix(df):
    labels = [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "Run-off",
        "uwr_rejection",
    ]  # sorted(df.recommended_category.unique().tolist())#
    migration_matrix = confusion_matrix(
        df.score_category_declared.astype(str), df.recommended_category.astype(str), labels=labels, normalize="true"
    )

    ax = sns.heatmap(migration_matrix, annot=True, cmap="Blues", fmt=".2%", cbar=False)

    ax.set_title("Migration Matrix\n\n")
    ax.set_xlabel("\nV6 rating")
    ax.set_ylabel("V5 rating ")

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    # Tilt x-axis labels by 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right")

    # Display the visualization of the Confusion Matrix.
    plt.tight_layout()
    os.makedirs(os.path.join("dvc_plots", "impact_analysis"), exist_ok=True)
    plt.savefig(os.path.join("dvc_plots", "impact_analysis", "migration_matrix.png"), dpi=600, bbox_inches="tight")
    plt.show()
    return


def plot_confusion_matrix2(df, normalization, filename):
    labels = [
        "PricingIneligibility",
        "EquifaxIneligibility",
        "uwr_rejection",
        "Pre-accepted",
        "inconsistent",
    ]  # sorted(df.recommended_category.unique().tolist())#
    migration_matrix = confusion_matrix(df.V5, df.V6, labels=labels, normalize=normalization)

    if normalization:
        ax = sns.heatmap(migration_matrix, annot=True, cmap="Blues", fmt=".2%", cbar=False)
    else:
        ax = sns.heatmap(migration_matrix, annot=True, cmap="Blues", cbar=False)

    ax.set_title("Migration Matrix\n\n")
    ax.set_xlabel("\nV6 ")
    ax.set_ylabel("V5  ")

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    # Tilt x-axis labels by 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right")

    # Display the visualization of the Confusion Matrix.
    plt.tight_layout()
    os.makedirs(os.path.join("dvc_plots", "impact_analysis"), exist_ok=True)
    plt.savefig(
        os.path.join("dvc_plots", "impact_analysis", "pre_acceptance_" + filename + ".png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()
    return
