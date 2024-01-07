import pathlib
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.calibration
import sklearn.inspection
import sklearn.metrics
import yaml
from yc_younipy.scorebands import compute_scoreband, compute_scoreband_binning_table

from utils import compute_global_figures, pprint_dataframe

sys.path.append("./src/pipeline_files")

sklearn.set_config(transform_output="pandas")

pd.set_option("display.max_rows", 200)
data = pathlib.Path("data")
preprocess = data / "preprocess"

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

target = params["target"]
test_size = params["validation_size"]
X0 = joblib.load(preprocess / "gmv" / "X.joblib")
X = X0.dropna(subset=(target,))
y = X[target].astype(float)

models_dir = data / "models"
version_cur = "v2.1"
version_new = "v3.1"
clf_cur = joblib.load(models_dir / f"{version_cur}.joblib")
clf_new = joblib.load(models_dir / f"{version_new}.joblib")


rows = []
clfs = [(version_cur, clf_cur), (version_new, clf_new)]


demands = joblib.load("data/preprocess/X.joblib").query("preapproval_reason != 'RejectionIneligibility'")
max_date = demands["application_date"].max() - pd.Timedelta(days=90)
demands = demands.loc[demands["application_date"].ge(max_date)]

pprint_dataframe(compute_global_figures(demands, target).hide(axis="index"))

for version, clf in clfs:
    optb, table = compute_scoreband_binning_table(
        clf, X, y, recalibrate=True, binning_params={"max_pvalue": 0.1}, use_proba=True
    )
    demands[version] = compute_scoreband(clf.predict_proba(demands)[:, 1], optb, table).values
df = demands.pivot_table(
    index=version_cur,
    columns=version_new,
    values="request_id",
    aggfunc="count",
    margins=True,
)
pprint_dataframe(df.style.format(thousands=",").background_gradient(axis=1))
segments = {}
df = pd.DataFrame(
    data={
        f"A1-A{segment}": [
            demands[version].isin([f"A{i}" for i in range(1, segment + 1)]).sum() for version, clf in clfs
        ]
        for segment in [2, 4, 6, 8]
    },
    index=[version for version, clf in clfs],
).T
df.index.name = "Segment"
df["delta"] = (df[version_new] - df[version_cur]) / df[version_cur]
pprint_dataframe(
    df.reset_index()
    .style.hide(axis="index")
    .format({"delta": "{:.0%}"}, thousands=",")
    .background_gradient(subset="delta", cmap="RdYlGn", vmin=-1, vmax=1)
)

demands[["request_id", version_cur, version_new]].melt(
    id_vars=["request_id"], var_name="Version", value_name="Band"
).pivot_table(index="Band", columns="Version", values="request_id", aggfunc="count").plot(kind="bar", figsize=(12, 4))
dvc_plots = pathlib.Path("dvc_plots") / "business_impacts"
dvc_plots.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(dvc_plots / "band-distribution.png")
plt.close("all")
