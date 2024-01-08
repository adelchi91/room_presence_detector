import pandas as pd
import os 
from alibi_detect.cd import ChiSquareDrift, TabularDrift
from alibi_detect.utils.saving import save_detector, load_detector

# Global variables 
FEATURES_COVARIATE = ["Temperature",  "Humidity", "Light", "CO2", "HumidityRatio"]


def import_data_new_data():
    # dataset with new data
    file_path = "./datatest2.txt"
    #file_path = "./datatraining.txt"
    # check if file exists 
    if os.path.exists(file_path):
        print(f"The file {file_path} exists.")
        df = pd.read_csv(file_path, index_col=0)
    else:
        print(f"The file {file_path} does not exist.")
        df = pd.DataFrame()
    return df

def import_original_data():
    df = pd.read_csv("datatraining.txt", index_col=0)
    # Defining covariate dataframe 
    X_ref = df[FEATURES_COVARIATE].copy()
    # drfit detector library - p_val is the p-value chosen for statistical meaning 
    cd = TabularDrift(p_val=.05, x_ref=X_ref.to_numpy(copy=True))
    return cd, df

def check_covariate_drift(df_new, df_original, cd):
    df_merged = pd.concat([df_original,df_new]).copy()
    preds = cd.predict(df_merged.to_numpy(copy=True))
    labels = ['No!', 'Yes!']
    print('Drift? {}'.format(labels[preds['data']['is_drift']]))
    # Save the drift detection result to a file
    result_filename = "drift_detection_result.txt"
    with open(result_filename, "w") as result_file:
        result_file.write(labels[preds['data']['is_drift']])
    return labels[preds['data']['is_drift']], df_merged

if __name__ == '__main__':
    cd, df_original = import_original_data()
    df_new = import_data_new_data()
    answer, df_merged = check_covariate_drift(df_new, df_original, cd)
    if answer=='Yes!':
        # scratch the training data with the updated data, so that a new model can be retrained
        df_merged.to_csv("datatraining.txt")
        print("data_training.txt was modified")
    print('ALL good mate')