import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score
import optbinning
import pathlib
from joblib import dump

# from utils.helpers import helpers
from utils.pipelines import grid, clf_rf, clf_v3_optb_logit

def import_data():
    # Open data
    df = pd.read_csv("datatraining.txt", index_col=0)
    # sort dataframe by date
    df = df.sort_values(by=['date'])
    # define covariate and target
    features_covariate = ["Temperature",  "Humidity", "Light", "CO2", "HumidityRatio"]
    X_train = df[features_covariate].copy()
    y_train = df["Occupancy"].copy()
    return X_train, y_train

def cross_validation(X_train, y_train):
    # Cross validation object definition and fit
    selection_metric = "roc_auc"
    cv = sklearn.model_selection.GridSearchCV(
        estimator=sklearn.pipeline.Pipeline(steps=[("clf", clf_v3_optb_logit)]),
        param_grid=grid,
        scoring={
            "neg_log_loss": "neg_log_loss",
            "roc_auc": "roc_auc",
            "brier": "neg_brier_score",
        },
        cv=sklearn.model_selection.StratifiedKFold(n_splits=5),
        verbose=1,
        refit=selection_metric,
        n_jobs=12,
        return_train_score=True,
    )

    cv.fit(X_train, y_train)
    return cv


if __name__ == '__main__':
    # import data
    X_train, y_train = import_data()
    # # cross-validation 
    # cv = cross_validation(X_train, y_train)
    # # save best model
    # filename = 'classification_model.joblib'
    # dump(cv.best_estimator_, filename)
    print('Helloo - all good my friend')









