import os
import sys

import joblib
import numpy as np
import yaml
from pandas import PeriodIndex, options

from helpers_local.helpers_factorizing import binning, plot_pop_over_time
from pipeline_files.helpers import v3_cat_features, v3_features  # NOQA

options.display.max_columns = 40
options.display.width = 1000

"""
Creation of temporal plots for all the variables of interest. This allows to verify the stability of each variable
in a temporal manner, which allows to identify if any possible temporal leakage might occur, or if a mapping for
specific variables might be necessary because of past changes operated by other teams. Usually these plots show
rather well whenever an UWR was created at a given time or whenever a new model was deployed.
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

# Load files
df_full = joblib.load(open(training_dataset_path, "rb"))

# Load params
params = yaml.safe_load(open("params.yaml"))
target = params["target"]


def convert_dataframe(alt_w_pred, vars, vars_cat):
    """
    This method is used to ensure correct dtype for each variable we wish to plot. Should not this be done,
    an error might occur when plotting the data.
    @param alt_w_pred: dataframe of interest
    @param vars: numerical variables
    @param vars_cat: categorical variables
    @return: dataframe with correct dtypes
    """
    for var in vars:
        alt_w_pred[var] = alt_w_pred[var].astype(np.float32)
    for var in vars_cat:
        alt_w_pred[var] = alt_w_pred[var].astype("string")
    return alt_w_pred


# definition of categorical variables - list
categorical_vars = v3_cat_features
# definition of numerical variables - list
numerical_vars = list(set(v3_features) - set(v3_cat_features))
# Other numerical variables of interest, not present in the model, that we wish to plot nonetheless
other_vars_num = [
    "remainder__partner_net_monthly_income",
    # "partner_income",
    # "freq_previously_funded_applications_per_days",
    # "number_of_applications_before_funding",
    # "number_of_previously_funded_applications",
]
# Other categorical of interest, not present in the model, that we wish to plot nonetheless
other_vars_cat = [
    # "number_of_paid_months_per_year",
    "remainder__number_of_salaries_per_year",
]

# ensuring correct dtypes
df_full_converted = convert_dataframe(df_full, numerical_vars, categorical_vars + other_vars_cat)
# convert application dates in semesters rather than months
df_full_converted["application_date"] = PeriodIndex(df_full_converted["application_date"], freq="Q-DEC").strftime(
    "%y-Q%q"
)
# Creating relevant bins for each feature, using the application date
output = binning(
    df_full_converted[v3_features + ["application_date", "contract_reference"] + other_vars_num + other_vars_cat],
    numerical_vars + other_vars_num,
    categorical_vars + other_vars_cat,
)

# Save outputs
os.makedirs(os.path.join("dvc_plots", "description_over_time"), exist_ok=True)
vars_ = output.filter(regex="^num__|^str__", axis=1).columns.to_list()
for var in vars_:
    for use_prop in [True, False]:
        bytes_plot = plot_pop_over_time(output[var], output["remainder__application_date"], "M", var, use_prop)
        with open(f"./dvc_plots/description_over_time/{var}_over_time_prop_{use_prop}.png", "wb") as f:
            f.write(bytes_plot)
