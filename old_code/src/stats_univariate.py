import os
import sys

import joblib
import yaml
from numpy import float32
from pandas import options
from yc_younipy.metrics.univariate.univariate_analysis import plot_univariate_analysis

from helpers_local.helpers_factorizing import binning
from pipeline_files.helpers import v3_cat_features, v3_features  # NOQA

# Definition of width and number of columns we wish to output in the pycharm terminal whenever printing a dataframe
options.display.max_columns = 40
options.display.width = 1000

"""
Creation of univariate plot that underline the relation between the default of interest and the variable used in the
model (or not). Be careful when jumping into conclusions, as these graphs do not point out interactions that might take
place between the remaining features.
"""

# verification that the correct number of arguments is being fed at this stage. The arguments are defined in dvc.yaml
# file.
if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython stats_model.py training_dataset_path validation_dataset_path pipeline_path variables_file"
    )
    sys.exit(1)

# Load files
training_dataset_path = sys.argv[1]
df_full = joblib.load(open(training_dataset_path, "rb"))

# Load params
params = yaml.safe_load(open("params.yaml"))
target = params["target"]


# Load variables set
categorical_vars = v3_cat_features
numerical_vars = list(set(v3_features) - set(v3_cat_features)) + [
    "simpleimputer-2__personal_age_difference",
    "simpleimputer-2__bank_age_difference",
    "remainder__mortgage_amount_difference",
    "remainder__rent_amount_difference",
]


def convert_dataframe(alt_w_pred, vars, vars_cat):
    """
    This method is used to ensure correct dtype for each variable we wish to plot. Should not this be done,
    an error might occur when plotting the data.
    @param alt_w_pred: dataframe of interest
    @param vars: numerical variables
    @param vars_cat: categorical variables
    @return:dataframe with correct dtypes
    """
    for var in vars:
        alt_w_pred[var] = alt_w_pred[var].astype(float32)
    for var in vars_cat:
        alt_w_pred[var] = alt_w_pred[var].astype("string")
    return alt_w_pred


# definition of categorical variables - list
categorical_vars = v3_cat_features
# definition of numerical variables - list
numerical_vars = list(set(v3_features) - set(v3_cat_features)) + [
    "simpleimputer-2__personal_age_difference",
    "simpleimputer-2__bank_age_difference",
    "remainder__mortgage_amount_difference",
    "remainder__rent_amount_difference",
]

# ensuring correct dtypes
df_full_converted = convert_dataframe(df_full, numerical_vars, categorical_vars)
# Creating relevant bins for each feature, using the application date
output = binning(
    df_full_converted[
        v3_features
        + [target, "contract_reference"]
        + [
            "simpleimputer-2__personal_age_difference",
            "simpleimputer-2__bank_age_difference",
            "remainder__mortgage_amount_difference",
            "remainder__rent_amount_difference",
        ]
    ],
    numerical_vars,
    categorical_vars,
)

# Save outputs
os.makedirs(os.path.join("dvc_plots", "stats_univariate_analysis"), exist_ok=True)
# variables on which we loop. The ouput dataframe renames the variables as num__<somestring>
vars_ = output.filter(regex="^num__|^str__", axis=1).columns.to_list()
# We drop rows where the target is not defined
output = output.dropna(subset="remainder__" + target).copy()
# We set the target dtype as a float64
output["remainder__" + target] = output["remainder__" + target].astype("float64")
# Creating the plots
for var in vars_:
    bytes_plot = plot_univariate_analysis(
        output[var], output["remainder__dn3_12"], missing_value="Missing"
    )  # , x_group=output['remainder__dataframe'])
    with open(f"./dvc_plots/stats_univariate_analysis/{var}_univariate.png", "wb") as f:
        f.write(bytes_plot)
