import pathlib
import warnings

import joblib
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import optbinning
import pandas as pd
import seaborn as sns
import sklearn
import tqdm
import yaml
from plotnine.utils import PlotnineWarning
from yc_younipy.preprocessing.most_frequent_binning import MostFrequentBinning
from yc_younipy.preprocessing.quantile_binning import QuantileBinning

from utils import compute_global_figures, is_notebook, pprint_dataframe

sklearn.set_config(transform_output="pandas")
sns.set_theme(style="whitegrid")
YC_PINK = "#C5A1FE"
YC_PINKS = matplotlib.colors.LinearSegmentedColormap.from_list("yc_pinks", [(1, 1, 1), YC_PINK])
figsize = (12, 4)
warnings.filterwarnings("ignore", category=PlotnineWarning)

directory = pathlib.Path(".")
data = directory / "data"
dvc_plots = directory / "dvc_plots" / "univariate"
dvc_plots.mkdir(parents=True, exist_ok=True)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
target = params["target"]
X = joblib.load(data / "preprocess" / "gmv" / "X.joblib")
pprint_dataframe(compute_global_figures(X, target))


X["application_date"] = X["application_date"].dt.tz_localize(None)
y = X[target].astype(float)
d = X["application_date"]
period = "1Q"


def plot_univariate_with_target(X, y, variable):
    plt.figure(figsize=figsize)
    ax = sns.pointplot(
        data=X.sort_values(variable).dropna(subset=(target,)),
        y=f"{variable}_bin",
        x=target,
        join=False,
    )
    plt.title(f"{target} w.r.t {variable}")
    plt.xlabel(f"Default rate ({target})")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    # plt.yticks(rotation=45)
    plt.xlim((0, 0.2))
    for label in ax.get_ymajorticklabels():
        label.set_ha("left")
    plt.tick_params(axis="y", direction="in", pad=-10)
    plt.tight_layout()


def plot_univariate_trend(X, variable, datecol, percent=False, colormap=None):
    data = (
        X.sort_values(variable)
        .groupby([pd.Grouper(key=datecol, freq="1Y"), f"{variable}_bin"], sort=False)["request_id"]
        .count()
        .unstack(level=-1)
        .fillna(0)
    )
    if percent:
        data = data.div(data.sum(axis=1), axis=0)
    ax = data.plot(kind="bar", stacked=True, figsize=figsize, colormap="RdYlGn")
    ax.set_xticklabels([x.strftime("%Y") for x in data.index], rotation=0)
    if percent:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    plt.xlabel("Application date")
    plt.tight_layout()


bar = tqdm.tqdm(total=len(params["variables"]))

for variable in params["variables"]:
    binning_type = variable["binning"]["type"]
    params = variable["binning"].get("params", {})
    key = variable["key"]
    bar.update()
    bar.set_description(f"Plotting univariate analysis for variable {key:32}")
    if binning_type == "quantile":
        X[f"{key}_bin"] = QuantileBinning(**params).fit_transform(X.convert_dtypes()[[key]]).iloc[:, 0]
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

    plot_univariate_with_target(X, y, key)
    if is_notebook():
        plt.show()
    plt.savefig(dvc_plots / f"{key}-target.png")

    plot_univariate_trend(X, key, "application_date")
    if is_notebook():
        plt.show()
    plt.savefig(dvc_plots / f"{key}-trend.png")

    plot_univariate_trend(X, key, "application_date", percent=True)
    if is_notebook():
        plt.show()
    plt.savefig(dvc_plots / f"{key}-trend-prop.png")

    plt.close("all")
bar.close()
