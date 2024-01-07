import pathlib
import sys

import joblib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import optbinning
import pandas as pd
import seaborn as sns
import sklearn.compose
import tqdm
import yaml
from sklearn.preprocessing import KBinsDiscretizer
from yc_younipy.preprocessing.most_frequent_binning import MostFrequentBinning
from yc_younipy.preprocessing.quantile_binning import QuantileBinning

from utils import compute_global_figures, is_notebook, pprint_dataframe

sys.path.append("./src/pipeline_files")
sns.set_theme(style="whitegrid")
YC_PINK = "#C5A1FE"
YC_PINKS = matplotlib.colors.LinearSegmentedColormap.from_list("yc_pinks", [(1, 1, 1), YC_PINK])

sklearn.set_config(transform_output="pandas")
data = pathlib.Path("data")
models_directory = data / "models"
dvc_plots_directory = pathlib.Path("dvc_plots") / "score_dependencies"
dvc_plots_directory.mkdir(exist_ok=True)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

target = params["target"]
X = joblib.load(data / "preprocess" / "gmv" / "X.joblib")
X[target] = X[target].astype(float)
pprint_dataframe(compute_global_figures(X, target))
X = X.dropna(subset=(target,))


bins = 5
clf_2_1 = joblib.load(models_directory / "v2.1.joblib")
clf_3_1 = joblib.load(models_directory / "v3.1.joblib")


discretizer = KBinsDiscretizer(bins, encode="ordinal")
X["v2.1"] = clf_2_1.predict_proba(X)[:, 1]
X["v3.1"] = clf_3_1.predict_proba(X)[:, 1]

X["v2.1"] = discretizer.fit_transform(X[["v2.1"]])["v2.1"]
X["v3.1"] = discretizer.fit_transform(X[["v3.1"]])["v3.1"]


versions = ("v2.1", "v3.1")


bar = tqdm.tqdm(total=len(params["variables"]))

for variable in params["variables"]:
    binning_type = variable["binning"]["type"]
    params = variable["binning"].get("params", {})
    key = variable["key"]
    bar.update()
    bar.set_description(f"Plotting score dependency for variable {key:32}")
    X = X.sort_values(key)
    if binning_type == "quantile":
        X[f"{key}_bin"] = QuantileBinning(**params).fit_transform(X.convert_dtypes()[[key]])[:, 0]
    elif binning_type == "cut":
        X[f"{key}_bin"] = "Missing"
        mask = ~X[key].isna()
        params["bins"] = [-np.inf] + params["bins"] + [np.inf]
        X.loc[mask, f"{key}_bin"] = pd.cut(X.loc[mask, key], **params).astype("string").astype(str)
    elif binning_type == "most_frequent":
        X[f"{key}_bin"] = MostFrequentBinning(**params).fit_transform(X.convert_dtypes()[[key]]).iloc[:, 0]
    elif binning_type == "optbinning":
        X[f"{key}_bin"] = optbinning.OptimalBinning(**params).fit_transform(
            X[key], X[target].astype(float), metric="bins"
        )
    else:
        X[f"{key}_bin"] = X[key].astype(str).fillna("Missing")
    fig, axes = plt.subplots(1, len(versions), figsize=(20, 5))

    for v, index in enumerate(versions):
        kwargs = {}
        if binning_type == "cut":
            kwargs["palette"] = "Greens"
        sns.pointplot(data=X, x=index, y=target, hue=f"{key}_bin", ax=axes[v], n_boot=1, **kwargs)
        axes[v].set_ylim((0, 0.2))
    plt.tight_layout()
    plt.savefig(dvc_plots_directory / f"{key}.png")
    if is_notebook():
        plt.show()
    plt.close()
bar.close()
