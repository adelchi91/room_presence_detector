import os
import pathlib
import sys

import joblib
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.pipeline
import yaml
from sklearn.utils import estimator_html_repr
from yc_younipy.metrics.model.roc_auc import gini_computation as gini

from utils import compute_global_figures, is_notebook, pprint_dataframe

sys.path.append("./src/")
from helpers_local.helpers_model_2_1 import v2_features  # NOQA

# libraries with the pipelines tested definition as well as some other methods used in this stage
from helpers_local.pipelines_cross_validation import clf_v2_1  # NOQA
from helpers_local.pipelines_cross_validation import clf_v2_2  # NOQA
from helpers_local.pipelines_cross_validation import clf_v2_3  # NOQA
from helpers_local.pipelines_cross_validation import clf_v3_optb_catboost  # NOQA
from helpers_local.pipelines_cross_validation import clf_v3_optb_forest  # NOQA
from helpers_local.pipelines_cross_validation import clf_v3_optb_logit  # NOQA
from helpers_local.pipelines_cross_validation import compute_feature_importance  # NOQA
from helpers_local.pipelines_cross_validation import get_split_end_dates  # NOQA
from helpers_local.pipelines_cross_validation import gini_per_split  # NOQA
from helpers_local.pipelines_cross_validation import grid  # NOQA
from helpers_local.pipelines_cross_validation import stats_per_split  # NOQA; NOQA

sys.path.append("./src/pipeline_files")
# Definition of variables for the models tested as well as any other type of other model/enconding method used
from helpers import preprocess_pipeline_model_3p1, v3_cat_features, v3_features  # NOQA

sklearn.set_config(transform_output="pandas")
sns.set_theme(style="whitegrid")

""""
The cross-validation step is were all the magic happens!
All the fitted methods/models are applied here in a methodological way, i.e. on each temporal split, so that no temporal
leakage is introduced.
All models of interest is tested by playing on the following:
1. type of model/encoding
2. variables choice
3. hyperparameters

This is achieved combining a gridsearch with a temporal split.

Finally several stats and graphs are shown to highlight the different improvements achieved by changing all the
different parameters, models...

The models tested are the following:
- Same model as in prod V2 already fitted (coefficients are fixed in LogisticRegressionV2_1 available in helpers
- Same model as V2_1, but retrained on more data with same variables, same encoding and same algorithm
- Same model as 2_2, but with an OptimalBinning encoding
- V3 includes new variables, i.e. v3_features
"""

# Colours definition for the ouput graphs
YC_PINK = "#C5A1FE"
YC_PINKS = matplotlib.colors.LinearSegmentedColormap.from_list("yc_pinks", [(1, 1, 1), YC_PINK])

# loading of parameters file
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# parameters
target = params["target"]
test_size = params["validation_size"]
# Output folder definition/creation
data = pathlib.Path("data")
preprocess = data / "preprocess"
dvc_plots = pathlib.Path("dvc_plots") / "cross_validation"
dvc_plots.mkdir(exist_ok=True)
cross_validation_dir = data / "cross_validation"
cross_validation_dir.mkdir(exist_ok=True)
os.makedirs(os.path.join("src", "pipeline_files"), exist_ok=True)

# Dataframes of interest
X = pd.read_pickle(data / "payloads_dataset.pkl")
X.rename(
    columns={"remainder__contract_reference": "contract_reference", "remainder__application_date": "application_date"},
    inplace=True,
)
# defaults
y = pd.read_pickle(data / "targets_dataset.pkl")
# retrieving the contract reference column needed to merge X and y, given that the request_id in risk table
# are inaccurate
X_expanded = pd.json_normalize(X["payload"], meta=["request_id"])
X["contract_reference"] = X_expanded["contract_reference"].copy()
X["application_date"] = X_expanded["application_date"].copy()
# Delete the DataFrame to free up memory
del X_expanded
# It is important to sort dataframe by dates in order for the temporal cross-validation to work correctly.
X = pd.merge(X, y, on="contract_reference", how="left").sort_values("application_date")
# perimeter of interest where the target is defined
X = X.dropna(subset=(target,))
y = X[target].astype(float)
# dataframe copy to use afterwards
X_full = X.copy()
# We need to select only payload and request id to send the data in the pipeline.
X = X[["request_id", "payload"]]
X_train = X.copy()
X_val = pd.DataFrame()
y_train = y.copy()
y_val = pd.Series()


# local functions
def print_cross_validation(cv):
    """
    prints cross-validation results in a dataframe, e.g. gini, brier, std...
    @param cv: cross-validation object
    @return: dataframe with relevant statistics
    """
    sort_key = f"mean_test_{cv.refit}" if isinstance(cv.refit, str) else "mean_test_score"

    df = (
        pd.json_normalize(cv.cv_results_["params"])
        .assign(
            **{
                f"{prefix}_{step}_{metric}": cv.cv_results_[f"{prefix}_{step}_{metric}"]
                for metric in cv.scoring
                for prefix in ("mean", "std")
                for step in ("test",)
            }
        )
        .sort_values(sort_key, ascending=False)
    )
    df["clf"] = df["clf"].apply(lambda clf: clf.version)
    try:
        df = df[~df["mean_test_gini"].isna()]
    except KeyError:
        df = df

    try:

        def apply_style(df):
            return (
                df.style.background_gradient(
                    axis=1,
                    vmin=0,
                    vmax=df["mean_test_gini"].max(),
                    subset="mean_test_gini",
                    cmap=YC_PINKS,
                )
                .format(precision=4)
                .hide(axis="index")
            )

        pprint_dataframe(apply_style(df))
        pprint_dataframe(
            apply_style(
                df.groupby("clf", as_index=False).first().sort_values(f"mean_test_{selection_metric}", ascending=False)
            )
        )
    except KeyError:
        df = df
    return df


# Pipelines names - pipelines are defined in a separate method that is imported at the beginning of this stage
clf_v2_1.version = "v2.1"
clf_v2_2.version = "v2.2"
clf_v2_3.version = "v2.3"
clf_v3_optb_logit.version = "v3.1-optb-logit"
clf_v3_optb_forest.version = "3.0-optb-forest"
clf_v3_optb_catboost.version = "3.0-catboost"

# Cross validation object definition and fit
selection_metric = "neg_log_loss"  # or brier or gini
cv = sklearn.model_selection.GridSearchCV(
    estimator=sklearn.pipeline.Pipeline(steps=[("clf", clf_v3_optb_logit)]),
    param_grid=grid,
    scoring={
        "neg_log_loss": "neg_log_loss",
        "gini": sklearn.metrics.make_scorer(gini, needs_proba=True),
        "brier": "neg_brier_score",
    },
    # "calib_error": sklearn.metrics.make_scorer(calibration_error, needs_proba=True),,
    cv=sklearn.model_selection.TimeSeriesSplit(n_splits=5),
    verbose=1,
    refit=selection_metric,
    n_jobs=12,
    return_train_score=True,
)
cv.fit(X_train, y_train)

X_train_transformed = preprocess_pipeline_model_3p1.fit_transform(X_train)
X_train_transformed.rename(
    columns={"remainder__contract_reference": "contract_reference", "remainder__application_date": "application_date"},
    inplace=True,
)
# X_train_transformed = cv.best_estimator_[0].fit_transform(X_train)
X_train_transformed = pd.merge(
    X_train_transformed, X_full[["contract_reference", target]], on="contract_reference", how="left"
)
# X_train_transformed[target] = y_train.copy()
print("# Train dataset")
pprint_dataframe(compute_global_figures(X_train_transformed, target).hide(axis="index"))
print("# Validation dataset")
if len(X_val > 0):
    pprint_dataframe(compute_global_figures(X_val, target).hide(axis="index"))
else:
    print("X_val is empty")

############################################
# plot of the best model gini perf splits. #
############################################
# Retrieving split dates
split_end_dates, split_end_dates_train = get_split_end_dates(
    X_train_transformed, X_train_transformed["application_date"], n_splits=5, cross_val_obj=cv
)
# Print the end date for each split
for i, end_date in enumerate(split_end_dates):
    print(f"Split {i + 1} End Date:", end_date)
gini_per_split(cv, split_end_dates)

###################
# stats_per_split #
###################
df_stats = stats_per_split(X_train_transformed, target, cv, split_end_dates)

######################################
# cross-validation results and plots #
######################################
df_cv = print_cross_validation(cv)
axes = (
    df_cv.query("clf__logistic__penalty=='l2'")
    .query("clf=='v3.1-optb-logit'")
    .groupby("clf__logistic__C")[[f"mean_test_{scoring}" for scoring in cv.scoring]]
    .mean()
    .plot(subplots=True, logx=True, figsize=(12, 8))
)
plt.tight_layout()
if is_notebook():
    plt.show()
plt.savefig(dvc_plots / "C.png")

# cross-validation best models candidates
# Create a boolean mask indicating the duplicate rows in the 'clf' column
mask = df_cv["clf"].duplicated(keep="first")
# Invert the boolean mask to select the non-duplicate rows
df_best_models = df_cv[~mask].copy()
df_best_models.to_csv(("./data/cross_validation/best_models.csv"))

############################################################
# Comparison between model in prod and new model candidate #
############################################################
rows = []
clfs = [
    ("v2.1", clf_v2_1, v2_features),
    ("v3.1", cv.best_estimator_, v3_features),
]

########################################
# Features importances GINI and  BRIER #
########################################
feature_importances_gini = compute_feature_importance(
    sklearn.metrics.make_scorer(gini, needs_proba=True), X_train, y_train, clfs[1:]
)
feature_importances_brier = compute_feature_importance("neg_brier_score", X_train, y_train, clfs[1:])

# print_shap_feature_importance(cv.best_estimator_[0], X_train_transformed)

# ablative df
values_to_find = ["v3.1-optb-logit"]
mask = df_cv["clf"].isin(values_to_find)

joblib.dump(cv, cross_validation_dir / "cv.joblib")
df_cv.to_csv(("./data/cross_validation/models_exploration.csv"))

df_stats.to_csv(cross_validation_dir / "df_splits_stats.csv")

models_dir = data / "models"
models_dir.mkdir(exist_ok=True)
clf_v2_1.fit(X, y)
joblib.dump(clf_v2_1, models_dir / "v2.1.joblib")
joblib.dump(cv.best_estimator_[0], cross_validation_dir / "v3.1.joblib")
cv.best_estimator_[0].fit(X, y)
joblib.dump(cv.best_estimator_[0], models_dir / "v3.1.joblib")
joblib.dump(cv.best_estimator_[0], models_dir / "v3.1.joblib")
# saving in pipeline_files for reproductibility purposes
# joblib.dump(cv.best_estimator_[0], os.path.join("src", "pipeline_files", "pipeline.joblib"))
joblib.dump(split_end_dates, "./data/cross_validation/split_end_dates.joblib")
joblib.dump(split_end_dates_train, "./data/cross_validation/split_end_dates_train.joblib")
# pipeline
with open(models_dir / "clf_v2_1.html", "w") as f:
    f.write(estimator_html_repr(clf_v2_1))
with open(models_dir / "clf_v2_2.html", "w") as f:
    f.write(estimator_html_repr(clf_v2_2))
with open(models_dir / "clf_v2_3.html", "w") as f:
    f.write(estimator_html_repr(clf_v2_3))
with open(models_dir / "clf_v3_optb_logit.html", "w") as f:
    f.write(estimator_html_repr(clf_v3_optb_logit))
