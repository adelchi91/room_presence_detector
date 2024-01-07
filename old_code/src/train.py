import os
import sys

import joblib
import optbinning
import pandas as pd
import sklearn.pipeline
import yaml
from sklearn.utils import estimator_html_repr
from yc_younipy.compose import ColumnsFilter
from yc_younipy.preprocessing.logging import FeaturesLogger

sys.path.append("./src/pipeline_files")
from helpers import preprocess_pipeline_model_3p1, v3_cat_features, v3_features  # NOQA

"""
This stage has a simple purpose: combining the preprocessing pipeline, which creates the features tested in the
cross-validation, with the pipeline created in the cross-validation for the best model retrieved.
As such we create pipeline object which is the ultimate pipeline that will be used in production, that does all the
steps together.
"""

# best params as found in cv.best_estimator_ obtained in the cross-validation stage
BEST_PARAMS = {"C": 0.0027825594022071257, "max_iter": 5000, "penalty": "l2", "solver": "saga", "random_state": 0}


if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py data/dataframe_for_analytics/X.joblib file")
    sys.exit(1)


# loading files
payloads_dataset_path = sys.argv[1]

# loading of parameters file
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# parameters
target = params["target"]
# Dataframes of interest
X = joblib.load(payloads_dataset_path)
y = X[target].astype(float)

# We need to select only payload and request id to send the data in the pipeline.
X = X[["request_id", "payload"]]
X_train = X.copy()
X_val = pd.DataFrame()
y_train = y.copy()
y_val = pd.Series()


# features of interest for model v3
v3_selector = ColumnsFilter(v3_features).set_output(transform="pandas")
# New final pipeline. The best model retrieved in the cross-validation pipeline is obtained by doing
# cv.best_estimator_

preprocessing_pipeline = sklearn.pipeline.Pipeline(
    steps=[
        (
            "preprocess",
            preprocess_pipeline_model_3p1,
        ),
        (
            "selector",
            v3_selector,
        ),
    ]
)

learning_pipeline = sklearn.pipeline.Pipeline(
    steps=[
        (
            "encoder",
            optbinning.BinningProcess(
                variable_names=v3_features,
                categorical_variables=v3_cat_features,
                max_pvalue=0.05,
            ),
        ),
        ("scaler", sklearn.preprocessing.StandardScaler()),
        (
            "logistic",
            sklearn.linear_model.LogisticRegression(**BEST_PARAMS),
        ),
    ]
)

pipeline = sklearn.pipeline.Pipeline(
    steps=[
        ("preprocessing_pipeline", preprocessing_pipeline),
        ("features_logger", FeaturesLogger()),
        ("learning_pipeline", learning_pipeline),
    ]
)

pipeline.fit(X_train, y_train)

# saving final pipeline
os.makedirs(os.path.join("dvc_plots", "train"), exist_ok=True)
os.makedirs(os.path.join("data", "train"), exist_ok=True)
os.makedirs(os.path.join("src", "pipeline_files"), exist_ok=True)
joblib.dump(pipeline, os.path.join("data", "train", "pipeline.joblib"))  # saving for analytics
joblib.dump(pipeline, os.path.join("src", "pipeline_files", "pipeline.joblib"))  # saving for prod
# saving pipeline as html file
with open("dvc_plots/train/clf_v3_optb_logit.html", "w") as f:
    f.write(estimator_html_repr(pipeline))
