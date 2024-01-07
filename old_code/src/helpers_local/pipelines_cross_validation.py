import pathlib
import sys

import catboost
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import optbinning
import pandas as pd
import sklearn.linear_model
import sklearn.pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from yc_younipy.compose import ColumnsFilter

sys.path.append("./src/")
from helpers_local.helpers_model_2_1 import (  # NOQA
    LogisticRegressionV2_1,
    OneHotEncoderV2_1,
    OptBinning2DEncoderV2,
    preprocess_pipeline_model_2p1,
    v2_features,
)

sys.path.append("./src/pipeline_files")
from pipeline_files.helpers import (  # NOQA
    preprocess_pipeline_model_3p1,
    v3_cat_features,
    v3_features,
)

YC_PINK = "#C5A1FE"
YC_PINKS = matplotlib.colors.LinearSegmentedColormap.from_list("yc_pinks", [(1, 1, 1), YC_PINK])

# Folders where files are saved
data = pathlib.Path("data")
dvc_plots = pathlib.Path("dvc_plots") / "cross_validation"
cross_validation_dir = data / "cross_validation"

# features of interest for model currently in prod, i.e. V2
v2_selector = ColumnsFilter(v2_features)
# features of interest for model v3
v3_selector = ColumnsFilter(v3_features)

# Pipelines definition for all of the model candidates
clf_v2_1 = sklearn.pipeline.Pipeline(
    steps=[
        (
            "preprocess",
            preprocess_pipeline_model_2p1,
        ),
        (
            "selector",
            v2_selector,
        ),
        (
            "encoder",
            OneHotEncoderV2_1(),
        ),
        (
            "logistic",
            LogisticRegressionV2_1(),
        ),
    ]
)

clf_v2_2 = sklearn.pipeline.Pipeline(
    steps=[
        (
            "preprocess",
            preprocess_pipeline_model_2p1,
        ),
        (
            "selector",
            v2_selector,
        ),
        (
            "encoder",
            sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        ),
        (
            "logistic",
            sklearn.linear_model.LogisticRegression(random_state=0, penalty="l2", solver="lbfgs", C=0.03),
        ),
    ]
)

clf_v2_3 = sklearn.pipeline.Pipeline(
    steps=[
        (
            "preprocess",
            preprocess_pipeline_model_2p1,
        ),
        (
            "encoder",
            OptBinning2DEncoderV2(),
        ),
        (
            "selector",
            v2_selector,
        ),
        (
            "logistic",
            sklearn.linear_model.LogisticRegression(random_state=0, penalty="l2", solver="lbfgs", C=0.01),
        ),
    ]
)

clf_v3_optb_forest = sklearn.pipeline.Pipeline(
    steps=[
        (
            "preprocess",
            preprocess_pipeline_model_3p1,
        ),
        (
            "selector",
            v3_selector,
        ),
        (
            "encoder",
            optbinning.BinningProcess(
                variable_names=v3_features,
                categorical_variables=v3_cat_features,
            ),
        ),
        (
            "forest",
            sklearn.ensemble.RandomForestClassifier(random_state=0),
        ),
    ]
)

clf_v3_optb_catboost = sklearn.pipeline.Pipeline(
    steps=[
        (
            "preprocess",
            preprocess_pipeline_model_3p1,
        ),
        (
            "selector",
            v3_selector,
        ),
        (
            "catboost",
            catboost.CatBoostClassifier(cat_features=v3_cat_features, random_state=0, verbose=False),
        ),
    ]
)

# Create the feature selection model
feature_selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=200, random_state=42), threshold=0.001)
feature_selector = SelectFromModel(
    sklearn.linear_model.LogisticRegression(
        C=0.01, l1_ratio=0.2, max_iter=5000, penalty="elasticnet", random_state=2023, solver="saga"
    )
)

clf_v3_optb_logit = sklearn.pipeline.Pipeline(
    steps=[
        (
            "preprocess",
            preprocess_pipeline_model_3p1,
        ),
        (
            "selector",
            v3_selector,
        ),
        (
            "encoder",
            optbinning.BinningProcess(
                variable_names=v3_features,
                categorical_variables=v3_cat_features,
                max_pvalue=0.05,
            ),
        ),
        # ("removing_features_with_low_variance", VarianceThreshold(threshold=0.01)),
        ("scaler", sklearn.preprocessing.StandardScaler()),
        # ('feature_selection', feature_selector),
        (
            "logistic",
            sklearn.linear_model.LogisticRegression(random_state=0),
        ),
    ]
)

# grid of hyperparameters to be tested
grid = [
    {"clf": [clf_v2_1]},
    # {
    #     "clf": [clf_v2_2],
    #     "clf__logistic__C": np.logspace(-3, 1, 5),
    #     "clf__logistic__solver": ["saga"],
    #     "clf__logistic__penalty": ["l2"],
    # },
    # {
    #     "clf": [clf_v2_3],
    #     "clf__logistic__C": np.logspace(-3, 1, 5),
    #     "clf__logistic__solver": ["saga"],
    #     "clf__logistic__penalty": ["l2"],
    # },
    {
        "clf": [clf_v3_optb_logit],
        # "clf__encoder__binning_fit_params": { [None, 0.01, 0.02, 0.03, 0.04, 0.05]},
        # "clf__logistic__C": [0.01, 0.03, 0.1, 0.3, 1],
        "clf__logistic__C": np.logspace(-3, 1, 10),  # np.linspace(0.001, 1, 10), # np.logspace(-3, 1, 5),
        "clf__logistic__solver": ["saga"],
        "clf__logistic__penalty": ["l1", "l2"],  # "elasticnet"],
        # "clf__logistic__l1_ratio": [0.0, 0.01, 0.1, 0.2, 0.5, 0.8, 1.0],
        "clf__logistic__max_iter": [5000],
    },
    # {
    #     "clf": [clf_v3_optb_logit],
    #     "clf__logistic__C": [0.01, 0.03, 0.1, 0.3, 1],
    #     "clf__logistic__solver": ["saga"],
    #     "clf__logistic__penalty": ["elasticnet"],
    #     "clf__logistic__l1_ratio": [.5, .75, .95, .98],
    # },
    # {
    #     "clf": [clf_v3_optb_forest],
    #     "clf__forest__criterion": ["gini", "entropy", "log_loss"],
    # },
    # {
    #     "clf": [clf_v3_optb_catboost],
    #     "clf__catboost__learning_rate": [0.01, 0.03],
    # }
]


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


def get_split_end_dates(X, dates, n_splits, cross_val_obj):
    tscv = cross_val_obj.cv  # sklearn.model_selection.TimeSeriesSplit(n_splits=n_splits)
    split_end_dates = []
    split_end_dates_train = []
    for train_index, test_index in tscv.split(X):
        test_end_date = dates.iloc[test_index[-1]]
        train_end_date = dates.iloc[train_index[-1]]
        split_end_dates.append(test_end_date)
        split_end_dates_train.append(train_end_date)

    return split_end_dates, split_end_dates_train


def stats_per_split(df, target, cv, split_end_dates):
    # filtering on the perimeter of interest, i.e. where the target is defined.
    df = get_scope_data(df.copy(), target)
    # Get the GINI scores and standard deviations for all splits of the best model
    test_gini_splits = []
    for i in range(len(split_end_dates)):
        split_gini = cv.cv_results_[f"split{i}_test_gini"][cv.best_index_]
        test_gini_splits.append(split_gini)

    default_rates = []
    n_defaults = []
    n_contracts = []

    for end_date in split_end_dates:
        # Filter df to get split
        split = df[df["application_date"] < end_date]

        # Calculate metrics
        n_defaults.append(split[target].sum())
        n_contracts.append(len(split))
        default_rate = n_defaults[-1] / n_contracts[-1]
        default_rates.append(default_rate)

    # Create results dataframe
    results = pd.DataFrame(
        {
            "end_date": split_end_dates,
            "default_rate": default_rates,
            "n_defaults": n_defaults,
            "n_contracts": n_contracts,
            "gini": test_gini_splits,
        }
    )
    return results


def gini_per_split(cv, split_end_dates):
    def plot_graph(test_gini_splits1, test_gini_splits2, split_end_dates, name="GINI"):
        # Get the number of splits
        num_splits = len(test_gini_splits)

        # Create an array of split indices
        split_indices = np.arange(1, num_splits + 1)
        # Format the split end dates to display only the date portion
        formatted_dates = [date.strftime("%Y-%m-%d") for date in split_end_dates]

        # Plot the GINI scores
        plt.figure(figsize=(8, 5))
        plt.plot(split_indices, test_gini_splits1, marker="o", linestyle="-", label="candidate_model")
        plt.plot(split_indices, test_gini_splits2, marker="o", linestyle="-", label="model 2.1")
        # Set the x-axis labels to split end dates
        plt.xticks(split_indices, formatted_dates, rotation=45)

        plt.xlabel("Split End Date")
        plt.ylabel(name)
        plt.title(f"{name} Scores ")
        plt.legend(loc="best")
        plt.savefig(dvc_plots / f"splits_perf_{name}.png", bbox_inches="tight")
        # plt.show()
        return

    def plot_graphs(test_gini_splits, modele, split_end_dates, name="GINI"):
        # Get the number of splits
        num_splits = len(test_gini_splits)

        # Create an array of split indices
        split_indices = np.arange(1, num_splits + 1)
        # Format the split end dates to display only the date portion
        formatted_dates = [date.strftime("%Y-%m-%d") for date in split_end_dates]

        # Plot the GINI scores
        plt.figure(figsize=(8, 5))
        plt.plot(split_indices, test_gini_splits, marker="o", linestyle="-")
        # Set the x-axis labels to split end dates
        plt.xticks(split_indices, formatted_dates, rotation=45)

        plt.xlabel("Split End Date")
        plt.ylabel(name)
        plt.title(f"{name} Scores for {modele}")
        plt.savefig(dvc_plots / f"{modele}_splits_perf_{name}.png", bbox_inches="tight")
        # plt.show()
        return

    # Get model tested list
    pipelines = cv.cv_results_["param_clf"]

    # Find index of clf_v2_1 model
    v2_1_index = [i for i, p in enumerate(pipelines) if p.version == "v2.1"][0]

    # Get the GINI scores and standard deviations for all splits of the best model
    test_gini_splits = []
    test_neg_log_loss_splits = []
    test_brier_splits = []
    test_gini_splits_v2_1 = []
    test_neg_log_loss_splits_v2_1 = []
    test_brier_splits_v2_1 = []
    for i in range(len(split_end_dates)):
        split_gini = cv.cv_results_[f"split{i}_test_gini"][cv.best_index_]
        split_neg_log_loss = cv.cv_results_[f"split{i}_test_neg_log_loss"][cv.best_index_]
        split_brier = cv.cv_results_[f"split{i}_test_brier"][cv.best_index_]

        split_gini_v2_1 = cv.cv_results_[f"split{i}_test_gini"][v2_1_index]
        split_neg_log_loss_v2_1 = cv.cv_results_[f"split{i}_test_neg_log_loss"][v2_1_index]
        split_brier_v2_1 = cv.cv_results_[f"split{i}_test_brier"][v2_1_index]

        test_gini_splits.append(split_gini)
        test_neg_log_loss_splits.append(split_neg_log_loss)
        test_brier_splits.append(split_brier)

        test_gini_splits_v2_1.append(split_gini_v2_1)
        test_neg_log_loss_splits_v2_1.append(split_neg_log_loss_v2_1)
        test_brier_splits_v2_1.append(split_brier_v2_1)
    # test_gini_splits = [
    #     cv.cv_results_['split0_test_gini'][cv.best_index_],
    #     cv.cv_results_['split1_test_gini'][cv.best_index_],
    #     cv.cv_results_['split2_test_gini'][cv.best_index_],
    #     cv.cv_results_['split3_test_gini'][cv.best_index_]
    # ]
    # plot
    plot_graph(test_gini_splits, test_gini_splits_v2_1, split_end_dates, "GINI")
    plot_graph(test_neg_log_loss_splits, test_gini_splits_v2_1, split_end_dates, "Logloss")
    plot_graph(test_brier_splits, test_gini_splits_v2_1, split_end_dates, "Brier")

    # plot_graph(test_gini_splits_v2_1, "v2_1", split_end_dates, "GINI")
    # plot_graph(test_neg_log_loss_splits_v2_1, "v2_1", split_end_dates, "Logloss")
    # plot_graph(test_brier_splits_v2_1, "v2_1", split_end_dates, "Brier")

    best_models = [
        "best_model1",
        "best_model2",
        "best_model3",
        "best_model4",
        "best_model5",
        "best_model6",
        "best_model7",
        "best_model8",
        "best_model9",
        "best_model10",
    ]
    # ranking of best models based on GINI performance
    gini_ranks = cv.cv_results_["rank_test_gini"]
    for idx, model in enumerate(best_models):
        other_best_indices = np.where(gini_ranks == (idx + 1))[0]
        # this is in case other_best_indices it en empty list
        # I think this might happen when there are two models with identical GINI performance.
        if len(other_best_indices) == 0:
            continue
        other_best_index = other_best_indices[0]
        test_gini_splits = []
        for i in range(len(split_end_dates)):
            split_gini = cv.cv_results_[f"split{i}_test_gini"][other_best_index]
            test_gini_splits.append(split_gini)
        # test_gini_splits = [
        #     cv.cv_results_['split0_test_gini'][other_best_index],
        #     cv.cv_results_['split1_test_gini'][other_best_index],
        #     cv.cv_results_['split2_test_gini'][other_best_index],
        #     cv.cv_results_['split3_test_gini'][other_best_index]
        # ]
        # plot
        plot_graphs(test_gini_splits, model, split_end_dates)


def compute_feature_importance(scoring, X, y, clfs):
    ret = []
    for version, clf, subfeatures in clfs:
        model = clf.named_steps["clf"]["logistic"]
        feature_names = model.feature_names_in_
        # reproducing preprocessing steps to feed correct dataframe to model
        X_preprocessed = clf.named_steps["clf"]["preprocess"].transform(X)
        X_preprocessed = clf.named_steps["clf"]["encoder"].transform(X_preprocessed[feature_names])
        X_preprocessed = clf.named_steps["clf"]["scaler"].transform(X_preprocessed)
        df = permutation_importance(
            model,  # Use the classifier from the nested pipeline
            X_preprocessed,
            y,
            random_state=0,
            scoring=scoring,
        )
        df = (
            pd.DataFrame(
                {
                    "importances_mean": df["importances_mean"],
                    "importances_std": df["importances_std"],
                },
                index=feature_names,
            )
            .sort_values("importances_mean", ascending=False)
            .rename_axis("variable")
            .head(50)
        )
        df["importances_cum"] = df["importances_mean"].cumsum() / df["importances_mean"].sum()
        print(f"{version} {scoring}")
        print()
        ret.append(df)
    df.to_csv(cross_validation_dir / f"feature_importance_permutation_{scoring}.csv")
    return ret


#
# def compute_feature_importance(scoring, X, y, clfs):
#     ret = []
#     for version, clf, subfeatures in clfs:
#         df = sklearn.inspection.permutation_importance(
#             clf,
#             X[subfeatures],
#             y,
#             random_state=0,
#             scoring=scoring,
#         )
#         df = (
#             pd.DataFrame(
#                 {
#                     "importances_mean": df["importances_mean"],
#                     "importances_std": df["importances_std"],
#                 },
#                 index=subfeatures,
#             )
#             .sort_values("importances_mean", ascending=False)
#             .rename_axis("variable")
#             .head(50)
#         )
#         df["importances_cum"] = df["importances_mean"].cumsum() / df["importances_mean"].sum()
#         print(f"{version} {scoring}")
#         style = df.style.bar(
#             axis=1,
#             vmin=df["importances_mean"].min(),
#             vmax=df["importances_mean"].max(),
#             subset="importances_mean",
#             color=YC_PINK,
#         )
#         # pprint_dataframe(style, showindex="always")
#         print()
#         ret.append(df)
#     df.to_csv(cross_validation_dir / f"feature_importance_permutation_{scoring}.csv")
#     return ret
#
#
# def print_shap_feature_importance(clf, X):
#     X = clf[:-1].transform(X)
#     explainer = shap.LinearExplainer(cv.best_estimator_.steps[-1][1].steps[-1][1], X)
#     shap_values = explainer.shap_values(X)
#     shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
#     plt.savefig(dvc_plots / "shap_summary_plot.png")
#
#     if is_notebook():
#         plt.show()
