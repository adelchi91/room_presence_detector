import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from api_gateway.payload_generation.payload_generation import generate_payload_template
from pandas import DataFrame, concat, isna, melt, merge, options, read_pickle
from waterfall_chart import plot as waterfall
from yc_datamarty.query import QueryHandler
from yc_datamarty.utils import build_payload_query

from analysis_libary import score_bands
from pipeline_files.helpers import (
    preprocess_pipeline_model_3p1 as pipeline_to_preprocess,
)
from pipeline_files.helpers import v3_cat_features, v3_features

"""
Score bands calculation on all and training data and measure the impact of the new candidate model with respect to the model in production.
Impact is measured on the recent demands:
- preacceptance on the different score bands
- matrix migration

in various configurations:
- by redefining the score bands calibration (what usually CPM does)
- by applying the most recent risk bands decided on the recent demand
"""

# %%
# YOU NEED TO ADAPT THESE VALUES TO YOUR NEEDS ! a
MY_CONTEXT = "toto"
MY_WORKFLOW = None
MY_BASE_DATASET = "yc-data-science.one_datamart"
MY_BASE_TABLE = "all"
BANK_READER_ENRICHMENT_OUTPUT_VERSION = None

# Instantiate BigQuery client
bq_runner = QueryHandler()

options.display.max_columns = 40
options.display.width = 1000
# %%
# verification that the correct number of arguments is being fed at this stage. The arguments are defined in dvc.yaml
# file.
if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython business_impact_analysis.py training_dataset_path pipeline_path variables_file")
    sys.exit(1)
# %%
# Load files
training_dataset_path = sys.argv[1]
pipeline_path = sys.argv[2]
pipeline_2p1_path = sys.argv[3]

# Load files
df_full = joblib.load(open(training_dataset_path, "rb"))
pipeline = joblib.load(open(pipeline_path, "rb"))
model = pipeline  # ['clf']
model_prod = joblib.load(open(pipeline_2p1_path, "rb"))

# Load params
params = yaml.safe_load(open("params.yaml"))
target = params["target"]

# Load variables set
cat_feature = v3_cat_features
form_variables = list(set(v3_features) - set(v3_cat_features))

# data typing
df_full = df_full.dropna(subset=(target,))
df_full[target] = df_full[target].astype("int")  # df_train[target] is here of type int64, which causes problems to
# scikit learn. I have to convert it in an int


# Save outputs
os.makedirs(os.path.join("data", "business_impact"), exist_ok=True)
os.makedirs(os.path.join("data", "business_impact", "payload_demands"), exist_ok=True)


# %%
def loading_demands():
    def update_verified_situation(row):
        if isinstance(row, dict):
            if "verified_situation" not in row:
                row["verified_situation"] = {}
            if not row["verified_situation"]:
                row["verified_situation"] = {"expenses": {"ongoing_credits": {"credit_lines": None}}}
            elif "expenses" not in row["verified_situation"]:
                row["verified_situation"]["expenses"] = {"ongoing_credits": {"credit_lines": None}}
            elif "ongoing_credits" not in row["verified_situation"]["expenses"]:
                row["verified_situation"]["expenses"]["ongoing_credits"] = {"credit_lines": None}
            else:
                row["verified_situation"]["expenses"]["ongoing_credits"]["credit_lines"] = None
        return row

    # def update_monthly_amount(row):
    #     if isinstance(row, dict):
    #         if 'verified_situation' not in row:
    #             row['verified_situation'] = {}
    #         if 'expenses' not in row['verified_situation']:
    #             row['verified_situation']['expenses'] = {}
    #         if 'ongoing_credits' not in row['verified_situation']['expenses']:
    #             row['verified_situation']['expenses']['ongoing_credits'] = {}
    #         credit_lines = row['verified_situation']['expenses']['ongoing_credits'].get('credit_lines', {})
    #         if credit_lines is None:
    #             credit_lines = {'monthly_amount': None}
    #         else:
    #             credit_lines.setdefault('monthly_amount', None)
    #         row['verified_situation']['expenses']['ongoing_credits']['credit_lines'] = credit_lines
    #     return row

    # Load configuration
    # config = json.load(open("./src/init/config.json", "rb"))
    # Load request_ids query
    select_rows_query = Path("src/sql/select_rows_demands_query.sql").read_text()
    # Load request_ids query
    select_targets_query = Path("src/sql/select_targets_demands_query.sql").read_text()
    # Build payload template without pydantic model
    # !!!! No need to regenerate payload template path. You can retrieve it frm before !!!!
    payload_template_path = generate_payload_template(  # noqa: C901
        context=MY_CONTEXT,
        output_folder=os.path.join("data", "business_impact", "payload_demands"),
        batch_size=1,
        pydantic_model_path=os.path.join("src", "payload_validation", "pydantic_model.py"),
        bank_reader_enrichment_output_version=BANK_READER_ENRICHMENT_OUTPUT_VERSION,
        workflow=MY_WORKFLOW,
    )
    with open(payload_template_path, "r") as f:
        payload_template = json.load(f)

    # Build payloads from datamart
    payloads_query = build_payload_query(
        query_handler=bq_runner,
        payload_template=payload_template,
        forced_aliases=[],  # config["forced_aliases"],
        forced_defaults=[],  # config["forced_defaults"],
        snapshot_path={"dataset": MY_BASE_DATASET, "table": MY_BASE_TABLE},
        temp_table_dataset="yc-data-science.trash",
        request_ids_query=select_rows_query,
    )
    X = bq_runner.execute(query=payloads_query)
    X["payload"] = X["payload"].apply(lambda x: json.loads(x))
    # stop here and create a new dvc step
    # Updating verified situation node with None values, for the business impact analysis on demand
    # (i.e. no verified infos)
    # This code checks if x is a dictionary (using isinstance(x, dict)), and if so, it creates a new dictionary by
    # copying x with the "verified_situation" node set to None. If x is not a dictionary, it leaves it as is.
    X["payload"] = X["payload"].apply(lambda x: {**x, "verified_situation": None} if isinstance(x, dict) else x)
    # X["payload"] = X["payload"].apply(lambda x: x.update({"verified_situation": None}))
    # I need the credit lines node
    X["payload"] = X["payload"].apply(update_verified_situation)
    # X['payload'] = X['payload'].apply(update_monthly_amount)
    # X["payload"] = X["payload"].apply(update_nested_dict)
    y = bq_runner.execute(query=select_targets_query)
    # DEMANDS
    X_transformed = pipeline.named_steps["preprocess"].transform(X)
    # removing duplicate contracts
    X_transformed.drop_duplicates(subset="request_id", keep="first", inplace=True)
    y.drop_duplicates(subset="request_id", keep="first", inplace=True)
    X_transformed = merge(X_transformed, y, on="request_id", how="right").sort_values("application_date")
    # removing application_dates that are null
    X_transformed = X_transformed.dropna(subset="application_date")
    # @Adelchi !! ATTENTION !! there is a problem with you SQL queries, given that missing score bands, which should be
    # Runoffs, have a preapproval_status of pre_approved. See line below
    print(X_transformed[lambda x: x.score_category_declared.isna()].preapproval_status.value_counts(dropna=False))
    print("Figures for the whole dataset")
    # pprint_dataframe(compute_global_figures(X, target).hide(axis="index"))
    return X, X_transformed


def runoff_and_uwr_rejection(training_data_raw):
    # define new column
    training_data_raw["score_category_detailed"] = training_data_raw["score_category_declared"].copy()
    # I consider pre-rejected applications as Run-offs. The values in preapproval_status are correct and were
    # cross-checked with the dashboard
    training_data_raw.loc[lambda x: x.preapproval_status == "pre_rejected", "score_category_detailed"] = "Run-off"
    # I only have Uncategorized in Portugal - they correspond to both UWRs rejections and Runoffs (i.e. pre-rejected)
    training_data_raw["score_category_detailed"] = training_data_raw["score_category_detailed"].replace(
        ["Uncategorized"], "Run-off"
    )
    training_data_raw.loc[
        lambda x: (x["preapproval_reason"] == "RejectionIneligibility") & (x["score_category_detailed"] == "Run-off"),
        "score_category_detailed",
    ] = "uwr_rejection"
    return training_data_raw


def data_loader_recent_demands():
    file_path = "data/business_impact/training_data_raw.pickle"
    file_path2 = "data/business_impact/X_transformed.pickle"
    if os.path.exists(file_path) and os.path.exists(file_path2):
        training_data_raw = read_pickle(file_path)
        X_transformed = read_pickle(file_path2)
    else:
        training_data_raw, X_transformed = loading_demands()
        training_data_raw.to_pickle(file_path)
        X_transformed.to_pickle(file_path2)
    # preprocessing
    # training_data_raw = runoff_and_uwr_rejection(training_data_raw)
    X_transformed = runoff_and_uwr_rejection(X_transformed)

    # Correct figures for UWR-rejection and Run-off, as seen on dashboard
    print("These are the figures on the dashboard, verified by Luca")
    # print(training_data_raw.score_category_declared.value_counts(dropna=False, normalize=True))
    print(X_transformed.score_category_declared.value_counts(dropna=False, normalize=True))
    # filtering data on scope of interest
    # We remove SONAE given that the pre-accepted SONAE are present in the table
    # but not the pre-rejected according to CPM. We would need to get them from another table.
    # df_preprocessed = training_data_raw[
    #     lambda x: (x.partner_code == "YOUNITED")
    #     & (x.business_provider_code != "SONAE")
    #     & (x.score_category_detailed != "uwr_rejection")
    # ]
    df_tranformed_preprocessed = X_transformed[
        lambda x: (x.partner_code == "YOUNITED")
        & (x.business_provider_code != "SONAE")
        & (x.score_category_detailed != "uwr_rejection")
    ]
    # applying same filters on training_data_raw
    training_data_raw = training_data_raw[
        lambda x: x.request_id.isin(df_tranformed_preprocessed.request_id.values.tolist())
    ]
    return training_data_raw, df_tranformed_preprocessed


def computation_score_bands_applied_on_recent_demands(df_full, df_preprocessed):
    # Computing score bands on the totality of training dataset, i.e. df_full
    optb, table = score_bands.compute_scoreband_binning_table(
        model, df_full, df_full[target], binning_params={"max_pvalue": 0.1}, recalibrate=True, use_proba=True
    )
    print(table.drop("Totals").drop(["Count (%)", "WoE", "IV", "JS"], axis=1))
    # saving table results
    table.to_csv(os.path.join("data", "business_impact", "score_bands_table.csv"))
    # Computing score bands on recent population, i.e. past 2023-01-01, using table and optb computed on df_full
    # recommended_category is the score band computed
    # score_category_declared is the score band on the model Vprod.1 in production - see SQL query
    df_preprocessed["recommended_category"] = score_bands.compute_scoreband(
        model.predict_proba(df_preprocessed)[:, 1], optb, table
    ).values
    df_preprocessed["score_Vnew_computed"] = model.predict_proba(df_preprocessed)[:, 1]
    return df_preprocessed


def count_plot_migration_comparison(dfv, condition="with_runoffs"):
    # bar plot
    # Create a new DataFrame that concatenates the two columns
    # df_new = concat([df_preprocessed['score_category_declared'], df_preprocessed['recommended_category']])
    pct = (
        (dfv.groupby(["value", "variab" "le"]).size() / dfv.groupby(["variable"]).size() * 100)
        .reset_index()
        .rename({0: "percent"}, axis=1)
    )
    if condition == "with_runoffs":
        ordre = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "Run-off", "uwr_rejection"]
    else:
        ordre = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]

    ax = sns.barplot(
        data=pct,
        x="value",
        hue="variable",
        y="percent",
        # order=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'Run-off', 'uwr_rejection'],
        order=ordre,
        palette={"Vnew": "tab:blue", "V_prod_iso_risk": "tab:pink"},
        # {"Vnew": "tab:blue", "Vprod": "tab:orange", "V_prod_iso_risk": "tab:pink"}
    )

    # Tilt x-axis labels by 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.f%%')
    for patch in ax.patches:
        x, y = patch.get_x() + patch.get_width() / 2, patch.get_height()
        ax.text(x, y, f"{y:.0f}%", ha="center", va="bottom", fontsize=8)

    print("hey")

    plt.legend(loc="upper left")
    # add labels and legend
    plt.xlabel("Category")
    plt.ylabel("Value")
    # plt.legend()
    os.makedirs(os.path.join("dvc_plots", "impact_analysis"), exist_ok=True)
    os.makedirs(os.path.join("data", "impact_analysis"), exist_ok=True)
    plt.savefig(
        os.path.join("dvc_plots", "impact_analysis", f"count_plot_migration_comparison_{condition}.png"),
        dpi=600,
        bbox_inches="tight",
    )
    # display the plot
    plt.show()
    return


def transform_dataframe(df_original):
    # Create a new DataFrame to store the transformed data
    # df_transformed = DataFrame(columns=["value", "Vprod", "V_prod_iso_risk", "Vnew"])
    df_transformed = DataFrame(columns=["value", "V_prod_iso_risk", "Vnew"])

    # Iterate over the rows in the original DataFrame
    for i in range(len(df_original)):
        row = df_original.iloc[i]
        value = row["value"]
        for j in range(i + 1, len(df_original)):
            next_row = df_original.iloc[j]
            next_value = next_row["value"]
            if next_value == "Run-off" or next_value == "uwr_rejection":
                break
            new_value = value + "-" + next_value
            # new_Vprod = df_original.loc[i:j, "Vprod"].sum()
            new_Vprod_is_risk = df_original.loc[i:j, "V_prod_iso_risk"].sum()
            new_Vnew = df_original.loc[i:j, "Vnew"].sum()
            new_row = DataFrame(
                [[new_value, new_Vprod_is_risk, new_Vnew]],
                # [[new_value, new_Vprod, new_Vprod_is_risk, new_Vnew]],
                columns=["value", "V_prod_iso_risk", "Vnew"],
            )
            df_transformed = concat([df_transformed, new_row])
    # Add the final rows for 'Run-off' and 'uwr_rejection'
    df_transformed = concat([df_transformed, df_original.iloc[-2:]])
    return df_transformed.set_index("value")


def table_volume_score_bands_groupes(dfv):
    # number/volume statistics
    vol = (dfv.groupby(["value", "variable"]).size()).reset_index().rename({0: "number"}, axis=1)
    # Group the filtered DataFrame by 'variable' and 'value', and sum the 'number' column
    grouped_df = vol.groupby(["variable", "value"])["number"].sum().reset_index()
    # Pivot the grouped DataFrame to have 'variable' as columns and 'value' as rows
    pivoted_df = grouped_df.pivot_table(index="value", columns="variable", values="number", fill_value=0)
    pivoted_df.to_csv(os.path.join("data", "impact_analysis", "volumes_score_bands.csv"))
    df_transformed = transform_dataframe(pivoted_df.reset_index())
    df_transformed.to_csv(os.path.join("data", "impact_analysis", "volumes_score_bands_grouped.csv"))
    return


def plot_waterfall(df):
    # formatting df with only columns of interest
    df = df[["score_category_detailed", "recommended_category", "V_prod_iso_risk"]].copy()
    df = df.rename(columns={"score_category_detailed": "Vprod", "recommended_category": "Vnew"})
    df["Vprod"] = df["Vprod"].replace({"Run-off": "PricingIneligibility"})
    df["V_prod_iso_risk"] = df["V_prod_iso_risk"].replace({"Run-off": "PricingIneligibility"})
    df["Vnew"] = df["Vnew"].replace({"Run-off": "PricingIneligibility"})
    # list of conditions
    prod_condition = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "uwr_rejection"]
    nb_of_uwrs = df[lambda x: x.V_prod_iso_risk.isin(["uwr_rejection"])].V_prod_iso_risk.shape[0]
    Vprod_V_prod_iso_risk_prod_difference = (
        df[lambda x: x.Vprod.isin(prod_condition)].Vprod.shape[0]
        - df[lambda x: x.V_prod_iso_risk.isin(prod_condition)].V_prod_iso_risk.shape[0]
    )
    nb_of_inconsistent = df[lambda x: (x.is_consistent == "inconsistent") & (x.Vprod.isin(prod_condition))].shape[0]
    ################
    # df_waterfall #
    ################
    df_waterfall = DataFrame(
        columns=["Vprod_eligible", "new_uwrs", "Vprod_new_score_bands", "consistent", "Vnew_eligible"]
    )
    # all production is everybody but runoffs (including inconsistent)
    df_waterfall.loc[0, "Vprod_eligible"] = df[lambda x: x.Vprod.isin(prod_condition)].Vprod.shape[0]
    # new UWRs : we substract the population here above the people falling in uwr_rejection
    # df_waterfall.loc[0, 'new_uwrs'] = df_waterfall.loc[0, 'Vprod_eligible'] - nb_of_uwrs
    df_waterfall.loc[0, "new_uwrs"] = -nb_of_uwrs
    # Vprod_new_score_bands: we substract the population here above the difference between Vprod-prod and Vprod_is_risk-prod
    # df_waterfall.loc[0, 'Vprod_new_score_bands'] = df_waterfall.loc[0, 'new_uwrs'] - Vprod_V_prod_iso_risk_prod_difference
    df_waterfall.loc[0, "Vprod_new_score_bands"] = -Vprod_V_prod_iso_risk_prod_difference
    # df_waterfall.loc[0, 'consistent'] = df_waterfall.loc[0, 'Vprod_new_score_bands'] - nb_of_inconsistent
    df_waterfall.loc[0, "consistent"] = -nb_of_inconsistent
    # df_waterfall.loc[0, 'New_prod_benchmark'] = df_waterfall.loc[0, 'consistent']
    # plot
    # column_names = list(df_waterfall.columns)
    # values = df_waterfall.values.flatten()
    # sns.set_style("whitegrid")
    # plt.figure(figsize=(10, 6))
    # plt.xticks(rotation=45)
    # sns.barplot(x=column_names, y=values)
    plt.figure(figsize=(8, 6))
    # Add value label on top of the 'net' column
    net_value = df_waterfall.sum(axis=1)
    waterfall(df_waterfall.columns, df_waterfall.values.tolist()[0], formatting="{:,.0f}")
    # plt.text(len(df_waterfall) - 4, net_value + 100, f'{net_value}', ha='center')
    # Add the additional bar on the right-hand side
    # plt.twinx()
    df_waterfall.loc[0, "Vnew_eligible"] = df[
        lambda x: x.Vnew.isin(["A1", "A2", "A3", "A4", "A5", "A6", "A7"])
    ].Vnew.shape[0]
    plt.bar(len(df_waterfall) + 3, df_waterfall["Vnew_eligible"], color="orange")
    # Add value labels on top of the bars
    for i, value in enumerate(df_waterfall.values.tolist()[0] + [int(net_value[0])]):
        if (i == 4) or (i == 5):
            plt.text(i, value + 100, f"{value}", ha="center")
    plt.title("Waterfall Plot")
    plt.xlabel("Category")
    plt.ylabel("Value")
    plt.savefig(os.path.join("dvc_plots", "impact_analysis", "waterfall.png"), dpi=600, bbox_inches="tight")
    plt.show()


def assign_risk_category(score_value):
    """
    Latest risk hypotheses
    @param score_value: current score value
    @return:
    """
    # you can retrieve the numbers below from
    # SELECT score_band, score_min, score_max
    # FROM `yuc - pr - risk.pricing.grids`
    # WHERE score_version = 'credit_score_pt/v2_1'
    # AND replay_version = 1
    # group by score_band, score_min, score_max
    if isna(score_value):
        return "Run-off"  # Handle None values: I checked and they all correspond to
        # preapproval_reason==PricingIneligibility
    elif 9550 <= score_value < 10000:
        return "A2"
    elif 9200 <= score_value < 9550:
        return "A5"
    elif 8200 <= score_value < 9200:
        return "A7"
    elif score_value < 8200:
        return "Run-off"
    else:
        return "problem"


def new_uwr(df, column_name, score):
    # score on 10 000 basis
    df["score_base_tenthousand"] = (1 - df[score].astype(float)) * 10000
    df["has_already_had_a_younited_loan"] = df["has_already_had_a_younited_loan"].fillna(False)

    conditions = [
        (
            (df["has_already_had_a_younited_loan"] is not True)
            & (df[column_name] != "Run-off")
            & ~(df["housing_code"].isin(["HOME_OWNERSHIP_WITH_MORTGAGE", "HOME_OWNERSHIP_WITHOUT_MORTGAGE"]))
            & (df["score_base_tenthousand"] < 8836)
        ).astype(bool),
    ]

    # conditions = [
    #     ((df["has_already_had_a_younited_loan"] != True) &
    #      (df["housing_code"].isin(["HOME_OWNERSHIP_WITH_MORTGAGE", "HOME_OWNERSHIP_WITHOUT_MORTGAGE"])) &
    #      (df["score_base_tenthousand"] > 8200)).astype(bool),
    #     ((df["has_already_had_a_younited_loan"] != True) &
    #      ~(df["housing_code"].isin(["HOME_OWNERSHIP_WITH_MORTGAGE", "HOME_OWNERSHIP_WITHOUT_MORTGAGE"])) &
    #      (df["score_base_tenthousand"] > 8836)).astype(bool),
    #     ((df["has_already_had_a_younited_loan"] == True) & (df["score_base_tenthousand"] > 7900)).astype(bool),
    # ]

    choices = [
        "uwr_rejection",
    ]

    df[column_name] = np.select(conditions, choices, default=df[column_name])
    return df


# df_full => training dataset
# df_preprocessed => recent demands
# df_tranformed_preprocessed  => recent demands preprocessed
# %%
# recent demands with appropriate filters
df_preprocessed, df_tranformed_preprocessed = data_loader_recent_demands()
# score bands calculation on recent demands - addition of recommended_category and score_Vnew_computed columns
df_preprocessed = computation_score_bands_applied_on_recent_demands(df_full, df_preprocessed)
###########################
# migration matrix plot ###
###########################
# %%
df_tranformed_preprocessed = merge(
    df_tranformed_preprocessed,
    df_preprocessed[["request_id", "recommended_category", "score_Vnew_computed"]],
    on="request_id",
    how="left",
).sort_values("application_date")
score_bands.plot_confusion_matrix(df_tranformed_preprocessed)
# dataframe with volumes
# creating V_prod_iso_risk column which corresponds to Vprod score bands with Q1 2023 score bands cutoffs
# Nicolas verified df_preprocessed here below. Column score_category_detailed is correct
# Computing score bands on the totality of training dataset, i.e. df_full
# %%
optb, table = score_bands.compute_scoreband_binning_table(
    model_prod,
    df_full.dropna(subset=(target,)),
    df_full[target],
    binning_params={"max_pvalue": 0.1},
    recalibrate=True,
    use_proba=True,
)
df_preprocessed["score_value_V_prod_iso_risk"] = model_prod.predict_proba(df_preprocessed)[:, 1]
df_preprocessed["V_prod_iso_risk"] = score_bands.compute_scoreband(
    model_prod.predict_proba(df_preprocessed)[:, 1], optb, table
).values
print("## Model 2.1")
print(table)  # .drop("Totals").drop(["Count (%)", "WoE", "IV", "JS"], axis=1))
# Applying latest score band hypotheses
# df_preprocessed['V_prod_iso_risk'] = df_preprocessed['preapproval_score_value'].apply(assign_risk_category)
# df_preprocessed.loc[lambda x: x['recommended_category'] == 'uwr_rejection', 'V_prod_iso_risk'] = 'uwr_rejection'

# Need to feed V_prod_iso_risk in df_transformed_preprocessed dataframe
# %%
df_tranformed_preprocessed = merge(
    df_tranformed_preprocessed,
    df_preprocessed[["request_id", "V_prod_iso_risk", "score_value_V_prod_iso_risk"]],
    on="request_id",
    how="left",
).sort_values("application_date")
# new uwrs
# We assign the contracts that are not falling into the uwr category for the model at iso risk (pink)
df_tranformed_preprocessed = new_uwr(
    df_tranformed_preprocessed, "score_category_detailed", score="preapproval_score_value"
)
# df_tranformed_preprocessed = new_uwr(df_tranformed_preprocessed, "V_prod_iso_risk", score="score_value_V_prod_iso_risk")
# We have the same contracts that will be considered in the same category for the PROD scenario (orange)
df_tranformed_preprocessed.loc[
    lambda x: x["score_category_detailed"] == "uwr_rejection", "V_prod_iso_risk"
] = "uwr_rejection"
# dfv = melt(df_tranformed_preprocessed[["score_category_detailed", "recommended_category", "V_prod_iso_risk"]])
# dfv["variable"] = dfv["variable"].replace({"score_category_detailed": "Vprod", "recommended_category": "Vnew"})
dfv = melt(df_tranformed_preprocessed[["recommended_category", "V_prod_iso_risk"]])
dfv["variable"] = dfv["variable"].replace({"recommended_category": "Vnew"})
####################################
# count_plot_migration_comparison ##
####################################
count_plot_migration_comparison(dfv, condition="with_runoffs")
count_plot_migration_comparison(dfv, condition="without_runoffs")
print(
    dfv[lambda x: x.variable == "Vnew"][lambda x: x.value != "Run-off"].shape[0]
    - dfv[lambda x: x.variable == "V_prod_iso_risk"][lambda x: x.value != "Run-off"].shape[0]
)
#################################
# volumes_score_bands_grouped ###
#################################
# %%
table_volume_score_bands_groupes(dfv)
####################
# Waterfall plot ###
####################
# plot_waterfall(df_preprocessed)

print("max Vprod", df_tranformed_preprocessed.groupby("score_category_declared")["preapproval_score_value"].max())
print("min Vprod", df_tranformed_preprocessed.groupby("score_category_declared")["preapproval_score_value"].min())
print("mean Vnew", df_tranformed_preprocessed.groupby("score_category_declared")["score_Vnew_computed"].mean())

# %%
# Saving dataframe
# df_tranformed_preprocessed.rename(columns={"score_category_detailed": "Vprod", "recommended_category": "Vnew"},
#                                   inplace=True)
df_tranformed_preprocessed.rename(columns={"recommended_category": "Vnew"}, inplace=True)
df_tranformed_preprocessed.to_pickle(os.path.join("data", "impact_analysis", "df_preprocessed.pkl"))
# df_stats = df_tranformed_preprocessed[['borrowed_amount', 'Vnew', 'V_prod_iso_risk']].groupby(
#     ['Vnew', 'V_prod_iso_risk']).agg(['sum', 'count'])
df_stats = DataFrame()
df_stats.to_csv(os.path.join("data", "impact_analysis", "df_stats.csv"))
# %%
# Saving
df_full = joblib.load(open(training_dataset_path, "rb"))
df_full["prediction_v3"] = model.predict_proba(df_full)[:, 1]
df_full.to_pickle(os.path.join("data", "impact_analysis", "df_full.pkl"))
