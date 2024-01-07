import pathlib
import sys

import pandas as pd
import yaml

# elements passed in cmd of dvc.yaml
if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython qa_analysis.py df_train df_val \n")
    sys.exit(1)

# Load args
df_train_path = sys.argv[1]
df_val_path = sys.argv[2]

# Load files
df_train = pd.read_pickle(df_train_path)
df_val = pd.read_pickle(df_val_path)

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# QA Analysis
# Do your QA analysis here

# Create directory for outputs
pathlib.Path.mkdir(pathlib.Path("data", "qa_analysis"), exist_ok=True)

# Save outputs
"""
with open(pathlib.Path("data", "qa_analysis", "qa_analysis_train.json"), 'w') as f:
    json.dump(qa_analysis_train, f)

with open(pathlib.Path("data", "qa_analysis", "qa_analysis_val.json"), 'w') as f:
    json.dump(qa_analysis_val, f)
"""
