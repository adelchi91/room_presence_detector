import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score
import optbinning
import pathlib
from joblib import dump
from sklearn.metrics import roc_curve, auc, brier_score_loss
import joblib
import matplotlib.pyplot as plt


# from utils.helpers import helpers
from utils.pipelines import grid, clf_rf, clf_v3_optb_logit


FEATURES_COVARIATE = ["Temperature",  "Humidity", "Light", "CO2", "HumidityRatio"]

def import_data():
    # Open data
    df = pd.read_csv("datatraining.txt", index_col=0)
    # sort dataframe by date
    df = df.sort_values(by=['date'])
    # define covariate and target
    features_covariate = FEATURES_COVARIATE
    X_train = df[features_covariate].copy()
    y_train = df["Occupancy"].copy()
    return X_train, y_train

def import_validation_data():
    df = pd.read_csv("./datatest.txt", index_col=0)
    return df

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

def performance_and_graphs(df, clf):
    y = df["Occupancy"]
    y_pred = clf.predict_proba(df[FEATURES_COVARIATE])[:, 1]

    # Calculate AUC using y_prob
    auc_score = roc_auc_score(y, y_pred)
    performance_dict = {
                            "auc": auc_score,
                            "brier": sklearn.metrics.brier_score_loss(y, y_pred),
                            "logloss": sklearn.metrics.log_loss(y, y_pred),
                            "count": y.count(),
                            "observations": y.sum(),
                            "occupancy_rate": y.mean(),
                        }
    df = pd.DataFrame([performance_dict])
    df.to_csv("./data/performance_stats.csv")
    return 


def plot_roc_curve(clf, df_train, df_val, features, save_path=None):
    # Train Data
    y_train = df_train["Occupancy"]
    y_train_pred = clf.predict_proba(df_train[features])[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Validation Data
    y_val = df_val["Occupancy"]
    y_val_pred = clf.predict_proba(df_val[features])[:, 1]
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred)
    roc_auc_val = auc(fpr_val, tpr_val)

    # Plot ROC curves for both Train and Validation data
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label=f'Train AUC = {roc_auc_train:.2f}')
    plt.plot(fpr_val, tpr_val, color='green', lw=2, label=f'Validation AUC = {roc_auc_val:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    # Save the figure with tight layout
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    return 






if __name__ == '__main__':
    # import data
    X_train, y_train = import_data()
    # import validation
    df_val = import_validation_data()
    # # cross-validation 
    cv = cross_validation(X_train, y_train)
    # save best model
    filename = 'classification_model.joblib'
    dump(cv.best_estimator_, filename)
    # Stats
    model_fname_ = 'classification_model.joblib'
    clf = joblib.load(model_fname_)
    performance_and_graphs(df_val, clf)
    # Roc curve
    plot_roc_curve(clf, pd.concat([X_train,y_train],axis=1), df_val, FEATURES_COVARIATE, save_path='plots/roc_curve.png')
    print('Helloo - all good my friend')









