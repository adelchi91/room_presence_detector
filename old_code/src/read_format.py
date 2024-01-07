import json
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython read_format.py data-file priority_file dtypes_file_path\n")
    sys.exit(1)

# Load dataset
raw_dataset_path = sys.argv[1]
priority_dataset_path = sys.argv[2]
dtypes_file_path = sys.argv[3]

raw_dataset = pd.read_feather(raw_dataset_path)
with open(priority_dataset_path) as f:
    priority = json.load(f)
with open(dtypes_file_path) as f:
    dtypes = json.load(f)

formatted_dataset_path = pathlib.Path("data", "read_format", "formatted_dataset.pkl")


def dynamically_rename(df: pd.DataFrame, priorities: dict):
    for k, v in priorities.items():
        if len(v) == 1:
            df[k] = df[v[0]]
        elif len(v) == 2:
            df[k] = np.where(df[v[0]].notnull(), df[v[0]], df[v[1]])
        elif len(v) == 3:
            df[k] = np.where(
                df[v[0]].notnull(),
                df[v[0]],
                np.where(df[v[1]].notnull(), df[v[1]], df[v[2]]),
            )
        else:
            raise ValueError(f"List of priorities for key {k} should contain one or two elements")
    return df


def format_dataset(df):
    # Apply the dtypes
    df_formatted = df.astype(dtype=dtypes)

    # Creating priority dictionary dynamically
    df_renamed = dynamically_rename(df=df_formatted, priorities=priority)

    return df_renamed


pathlib.Path.mkdir(pathlib.Path("data", "read_format"), exist_ok=True)

formatted_dataset = format_dataset(raw_dataset)

# Save pickle files
with open(pathlib.Path(formatted_dataset_path), "wb") as f:
    pickle.dump(formatted_dataset, f)
