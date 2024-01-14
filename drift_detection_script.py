import pandas as pd
import os 
from alibi_detect.cd import ChiSquareDrift, TabularDrift
from alibi_detect.utils.saving import save_detector, load_detector
import joblib
from sklearn.metrics import roc_curve, auc, brier_score_loss


# Global variables 
FEATURES_COVARIATE = ["Temperature",  "Humidity", "Light", "CO2", "HumidityRatio"]
TARGET = "Occupancy"
MODEL_DRIFT_TOL=0.02 # i.e. two auc pts


def import_model():
    model_fname_ = 'classification_model.joblib'
    model = joblib.load(model_fname_)
    return model

def import_data_new_data():
    # dataset with new data
    file_path = "./datatest2.txt" # then drift
    #file_path = "./datatraining.txt" # then no-drift
    # check if file exists 
    if os.path.exists(file_path):
        print(f"The file {file_path} exists.")
        df = pd.read_csv(file_path, index_col=0)
    else:
        print(f"The file {file_path} does not exist.")
        df = pd.DataFrame()
    return df

def import_validation_data():
    df = pd.read_csv("./datatest.txt", index_col=0)
    return df

def import_original_data():
    df = pd.read_csv("datatraining.txt", index_col=0)
    # Defining covariate dataframe 
    X_ref = df[FEATURES_COVARIATE].copy()
    # drfit detector library - p_val is the p-value chosen for statistical meaning 
    cd = TabularDrift(p_val=.05, x_ref=X_ref.to_numpy(copy=True))
    return cd, df

def check_predictions_drift(df_new, df_original, model):
    predictions_original = model.predict(df_original[FEATURES_COVARIATE])
    predictions_new = model.predict(df_new[FEATURES_COVARIATE])
    # drfit detector library - p_val is the p-value chosen for statistical meaning 
    cd = TabularDrift(p_val=.05, x_ref=predictions_original)
    preds = cd.predict(predictions_new)
    labels = ['No!', 'Yes!']
    print('Drift on predictions? {}'.format(labels[preds['data']['is_drift']]))
    are_predictions_drifting = labels[preds['data']['is_drift']]
    return are_predictions_drifting

def check_covariate_drift(df_new, df_original, cd):
    df_merged = pd.concat([df_original,df_new]).copy()
    preds = cd.predict(df_merged[FEATURES_COVARIATE].to_numpy(copy=True))
    labels = ['No!', 'Yes!']
    print('Drift on covariate? {}'.format(labels[preds['data']['is_drift']]))
    # Save the drift detection result to a file
    result_filename = "drift_detection_result.txt"
    with open(result_filename, "w") as result_file:
        result_file.write(labels[preds['data']['is_drift']])
    return labels[preds['data']['is_drift']], df_merged

def check_auc(clf, df_val, dataset="validation"):
    y_probabilities = clf.predict_proba(df_val[FEATURES_COVARIATE])[:, 1]
    y_test = df_val[TARGET]
    fpr, tpr, _ = roc_curve(y_test, y_probabilities)
    roc_auc = auc(fpr, tpr)
    print(f"AUC {dataset}={roc_auc}")
    return roc_auc

if __name__ == '__main__':
    model = import_model()
    cd, df_original = import_original_data()
    df_validation = import_validation_data()
    df_new = import_data_new_data()
    auc_validation = check_auc(model, df_validation, "validation")
    auc_new = check_auc(model, df_new, "new")
    performance_drift = auc_validation - auc_new
    answer_drift_covariate, df_merged = check_covariate_drift(df_new, df_original, cd)
    answer_drift_predictions = check_predictions_drift(df_new, df_original, model)
    
    result_retraining = "drift_detection_result.txt"
    with open(result_retraining, "w") as result_file:
            result_file.write('No!')
    # check if there was a drift 
    if (answer_drift_covariate=='Yes!') | (answer_drift_predictions=='Yes'):
        # check if model performance is drifting
        if performance_drift>=MODEL_DRIFT_TOL:
        # Save the drift detection result to a file
            with open(result_retraining, "w") as result_file:
                result_file.write('Yes!')
            # scratch the training data with the updated data, so that a new model can be retrained
            df_merged.to_csv("datatraining.txt")
            print("data_training.txt was modified")
    else:
        print('answer is no')
    print('ALL good mate')