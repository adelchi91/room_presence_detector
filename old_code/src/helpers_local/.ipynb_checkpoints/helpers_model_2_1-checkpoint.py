import numpy as np
import optbinning
import pandas as pd
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.pipeline import Pipeline

# Constants:
FEATURES_TO_ENCODE_2_1 = [
    "debt_expenses_ratio_grp",
    "income_x_housing_grp",
    "prof_status_grp",
    "housing_status_x_age_grp",
    "marital_status_grp",
]
FEATURES_2_1 = [
    "debt_expenses_ratio_grp_no_expenses",
    "debt_expenses_ratio_grp_under_70_percent",
    "income_x_housing_grp_Group_2",
    "income_x_housing_grp_Group_3",
    "income_x_housing_grp_Group_4",
    "prof_status_grp_Group_2",
    "prof_status_grp_Group_3",
    "housing_status_x_age_grp_Group_2",
    "housing_status_x_age_grp_Group_3",
    "marital_status_grp_SINGLE",
]

HYPERPARAMETERS_2_1 = {
    "penalty": "none",
    "dual": False,
    "tol": 0.0001,
    "C": 1000000000.0,
    "fit_intercept": True,
    "intercept_scaling": 1,
    "class_weight": None,
    "random_state": 12456,
    "solver": "newton-cg",
    "max_iter": 100,
    "multi_class": "auto",
    "verbose": 0,
    "warm_start": False,
    "n_jobs": None,
    "l1_ratio": None,
}

COEFFICIENTS_2_1 = [
    0.3479,
    0.1884,
    0.5514,
    0.7195,
    1.4147,
    0.4974,
    0.6646,
    0.4182,
    0.254,
    0.5517,
]

VARIABLES_TO_COALESCE = {
    "banking.bank_code": "bank_code",
    "expenses.mortgage_amount": "mortgage_amount",
    "housing_situation.housing_code": "housing_code",
    "incomes.main_net_monthly_income": "main_net_monthly_income",
    "incomes.partner_net_monthly_income": "partner_net_monthly_income",
    "expenses.rent_amount": "rent_amount",
    "incomes.number_of_salaries_per_year": "number_of_salaries_per_year",
    "applicant.employment_situation.sector_code": "sector_code",
    "address.postal_code": "postal_code",
    "applicant.employment_situation.profession_code": "profession_code",
    "applicant.employment_situation.professional_situation_code": "professional_situation_code",
    "applicant.personal.cell_phone_number": "cell_phone_number",
    "applicant.personal.email": "personal.email",
    # Add more mappings as needed
}

# lambda functions cannot be pickled, and the Joblib library used for serialization relies on pickling to store objects.
# To overcome this issue, you can replace the lambda functions in your IMPUTED_METHODS_NUM dictionary with regular
# functions or methods.

# personal age has few values with missing data. This is a data
# quality problem present in the datamart. The occurences are less than 1%. Instead of dropping such records,
# we fill them with a the mean value.
VARIABLES_WITH_MEAN_VALUES_IMPUTATION = ["personal_age", "bank_age", "professional_age"]
VARIABLES_WITH_ZERO_VALUE_IMPUTATION = [
    "personal_age_difference",
    "bank_age_difference",
    "professional_age_difference",
    "ongoing_credits_amount",
    "rent_amount",
    "mortgage_amount",
]
# some missing email addresses are present. Less than 1% in total. We fill missing value with most frequent value
VARIABLES_WITH_MOST_FREQUENT_VALUE_IMPUTATION = ["email_domain", "phone_prefix"]

MOST_FREQUENT_EMAIL_DOMAINS = ["gmail.com", "hotmail.com", "sapo.pt", "live.com.pt", "outlook.pt", "outlook.com"]


class FlattenTransformer(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """
    This class is used in order to create the features used for model candidate v3
    - fit method is a dummy method.
    - transform is used for the features creation.
    """

    def __init__(self):
        self._is_fitted = False

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def transform(self, X_input, y=None):
        # not used at the moment
        def correlations_expansion(df):
            """
            Method not used at  the moment, but might be useful in future development when correlations will be taken
            into consideration
            @param df: dataframe with correlations node
            @return: Expanded dataframe
            """
            # select variables of interest
            df = df[["correlations", "request_id"]].copy()
            # Exploding field correlations
            expanded_df = df.explode("correlations").assign(
                is_granted=lambda x: x["correlations"].apply(
                    lambda c: c["application_status"]["granting"]["decision_status"] == "granted"
                    if isinstance(c, dict)
                    else None
                )
            )
            # Reset the index to ensure consistent merging later
            expanded_df = expanded_df.reset_index(drop=True)
            # Merge the calculated values back into the original dataframe
            df = df.merge(expanded_df, on="request_id", how="left")
            return df

        def credit_lines_dataframe_expansion(df, data_type="verified_situation"):
            # select variables of interest
            df = df[[f"{data_type}.expenses.ongoing_credits.credit_lines", "request_id"]].copy()
            # Exploding credit lines: creating a new dataframe with the exploded values
            expanded_df = df.explode(f"{data_type}.expenses.ongoing_credits.credit_lines")
            # Reset the index to ensure consistent merging later
            expanded_df = expanded_df.reset_index(drop=True)
            return expanded_df

        def create_ongoing_credits_amount(df, expanded_df, data_type="verified_situation"):
            # Filling None and NaN values with empty dictionaries. pd.json_normalize() function expects a list of
            # dictionaries as the input data. Whenever the value is None or NaN they are treated as float objects
            # causing the error when trying to access the values attribute of a float.
            filtered_data = expanded_df[f"{data_type}.expenses.ongoing_credits.credit_lines"].fillna({})

            # Flatten the nested JSON data using json_normalize
            flattened_df = pd.json_normalize(
                filtered_data,
                sep=".",
                errors="ignore",
                # record_path=["contract_reference"],
                # meta=[["contract_reference"]]
            )

            # Replace missing "monthly_amount" values with 0
            flattened_df["monthly_amount"].fillna(0, inplace=True)

            # Add the "contract_reference" column to the flattened dataframe
            flattened_df = pd.concat([expanded_df["request_id"], flattened_df], axis=1)

            # Group by any necessary columns and calculate the sum of monthly amounts
            if data_type == "verified_situation":
                grouped_df = flattened_df.groupby("request_id").agg(ongoing_credits_amount=("monthly_amount", "sum"))
            else:
                grouped_df = flattened_df.groupby("request_id").agg(
                    ongoing_credits_amount_declared=("monthly_amount", "sum")
                )

            # contract_reference was set as index in grouped_df, so we reset the index to apply the merge
            grouped_df = grouped_df.reset_index()

            # Merge the calculated values back into the original dataframe
            df = df.merge(grouped_df, on="request_id", how="left")
            return df

        def create_ongoing_credits_specific(df, expanded_df, specific="CARLOAN"):
            # Perform vectorized operations on the exploded dataframe
            dict_mapping = {
                "CARLOAN": "car",
                "PERSONALLOAN": "personal",
                "OTHER": "other",
                "LIQUIDITY": "liquidity",
                "REALESTATELOAN": "realestate",
                "DEBTS": "debts",
            }

            expanded_df[f"ongoing_credits_{specific.lower()}"] = expanded_df[
                "verified_situation.expenses.ongoing_credits.credit_lines"
            ].apply(
                lambda credit: credit.get("ongoing_credit_type_code") == specific if isinstance(credit, dict) else False
            )

            # Step 3: Group by any necessary columns and check if any credit is a CARLOAN
            grouped_df = expanded_df.groupby("request_id")[f"ongoing_credits_{specific.lower()}"].any().reset_index()
            # Rename the column based on dict_mapping
            value = dict_mapping.get(specific.upper(), specific.lower())
            grouped_df = grouped_df.rename(columns={f"ongoing_credits_{specific.lower()}": f"ongoing_credits_{value}"})
            # Step 4: Merge the results back into the original dataframe
            df = df.merge(grouped_df, on="request_id", how="left")
            return df

        def age_in_year(beg, end):
            return (end - beg) / np.timedelta64(1, "Y")

        # not used at the moment
        def diff_in_time(beg, end, unit="D"):
            return (end - beg) / np.timedelta64(1, unit)

        def coalesce_situation(path):
            return X[f"verified_situation.{path}"].combine_first(X[f"declared_situation.{path}"])

        # saving a copy, that will be used later
        # The transform method of a transformer should not modify the input data in place. Instead, it should return a
        # new DataFrame or array with the transformed features. Since your custom transformer modifies X in place, this
        # can lead to inconsistencies between different parts of your pipeline and cross-validation process.
        X = X_input.copy()
        X_temp = X_input.copy()
        # Exploding credit lines
        expanded_df = credit_lines_dataframe_expansion(X, data_type="verified_situation")
        # ongoing_credits_amount
        X = create_ongoing_credits_amount(X, expanded_df, data_type="verified_situation")
        # specific credits
        X = create_ongoing_credits_specific(X, expanded_df, specific="CARLOAN")
        X = create_ongoing_credits_specific(X, expanded_df, specific="PERSONALLOAN")
        X = create_ongoing_credits_specific(X, expanded_df, specific="OTHER")
        X = create_ongoing_credits_specific(X, expanded_df, specific="LIQUIDITY")
        X = create_ongoing_credits_specific(X, expanded_df, specific="REALESTATELOAN")
        X = create_ongoing_credits_specific(X, expanded_df, specific="DEBTS")
        # differences in credit declarations/verifications
        # Exploding credit lines - declared data
        expanded_df_declared = credit_lines_dataframe_expansion(X_temp, data_type="declared_situation")
        # ongoing_credits_amount - declared data
        X_temp = create_ongoing_credits_amount(X_temp, expanded_df_declared, data_type="declared_situation")
        X = X.merge(X_temp[["request_id", "ongoing_credits_amount_declared"]], on="request_id", how="left")
        X = X.assign(
            ongoing_credits_amount_difference=lambda x: x["ongoing_credits_amount_declared"].fillna(0)
            - x["ongoing_credits_amount"].fillna(0)
        )

        # other variables of interest
        application_date = pd.to_datetime(X["application_date"], utc=True)
        date_of_birth = pd.to_datetime(X["declared_situation.applicant.personal.date_of_birth"], utc=True)
        bank_account_opening_date = pd.to_datetime(coalesce_situation("banking.bank_account_opening_date"), utc=True)
        professional_situation_start_date = pd.to_datetime(
            coalesce_situation("applicant.employment_situation.professional_situation_start_date"),
            utc=True,
        )
        # The coalesce method is leveraged in order to consider the verified variable first, whenever present, or
        # the declared info otherwise.
        for input_path, output_name in VARIABLES_TO_COALESCE.items():
            X[output_name] = coalesce_situation(input_path)
        # columns to keep
        filtered_columns = X.filter(like="ongoing_credits", axis=1)
        filtered_columns = [col for col in filtered_columns if "." not in col]
        keep_cols = ["contract_reference", "request_id"] + filtered_columns + list(VARIABLES_TO_COALESCE.values())
        # Variables renaming and manipulation
        X = X[keep_cols].assign(
            # The following variables are basically renaming of nested variables with the corresponding sensible name
            # otherwise they correspond to age computations or differences between declared and verified variables.
            partner_code=X["business_context.partner_code"],
            business_provider_code=X["business_context.business_provider_code"],
            application_date=application_date,
            preapproval_status=X["application_status.preapproval.preapproval_status"],
            # borrowed_amount=X["product.borrowed_amount"],
            marital_status_code=coalesce_situation("marital_situation.marital_status_code"),
            #############################
            # age computations in years #
            #############################
            personal_age=age_in_year(date_of_birth, application_date),
            bank_age=age_in_year(bank_account_opening_date, application_date),
            professional_age=age_in_year(professional_situation_start_date, application_date),
            ##############################################################
            # filling missing values and renaming of specific modalities #
            ##############################################################
            professional_situation_code=X["professional_situation_code"].replace({"PENSION": "PENSIONER_RETIRED"}),
            number_of_salaries_per_year=X["number_of_salaries_per_year"].fillna(12),
            ###############
            # differences #
            ###############
            personal_age_difference=age_in_year(
                pd.to_datetime(X["declared_situation.applicant.personal.date_of_birth"], utc=True), application_date
            )
            - age_in_year(
                pd.to_datetime(X["verified_situation.applicant.personal.date_of_birth"], utc=True), application_date
            ),
            bank_age_difference=age_in_year(
                pd.to_datetime(X["declared_situation.banking.bank_account_opening_date"], utc=True), application_date
            )
            - age_in_year(bank_account_opening_date, application_date),
            mortgage_amount_difference=X["declared_situation.expenses.mortgage_amount"].fillna(0)
            - X["mortgage_amount"].fillna(0),
            main_net_monthly_income_difference=X["declared_situation.incomes.main_net_monthly_income"].fillna(0)
            - X["main_net_monthly_income"],
            professional_age_difference=age_in_year(
                pd.to_datetime(
                    X["declared_situation.applicant.employment_situation.professional_situation_start_date"], utc=True
                ),
                application_date,
            )
            - age_in_year(professional_situation_start_date, application_date),
            rent_amount_difference=X["declared_situation.expenses.rent_amount"]
            - coalesce_situation("expenses.rent_amount"),
            # profession_code=coalesce_situation("applicant.employment_situation.profession_code"),
            #################
            # New variables #
            #################
            postal_region=X["postal_code"].str[0:2],
            # we deliberately choose to use the verified cell_phone_number information
            phone_prefix=X["cell_phone_number"].str[0:6],
            # we deliberately choose to use the verified email information
            email_domain=X["personal.email"].str.replace(".*@", "", regex=True).str.lower(),
            # installment=X["product.borrowed_amount"] / X["product.maturity_in_months"],
            # maturity_in_months=X["product.maturity_in_months"],
            ongoing_credits_count=X["verified_situation.expenses.ongoing_credits.credit_lines"].apply(
                lambda credits: len(credits)
            ),
            # has_coborrower=X["declared_situation.co_applicant.has_co_borrower"],  # info only in declared info
        )
        # removal of observations with missing infos on borrowed amount and maturity
        # X["borrowed_amount"] = X["borrowed_amount"].fillna(10000)  # X.dropna(subset=["borrowed_amount"])
        # X["maturity_in_months"] = X["maturity_in_months"].fillna(60)  # X.dropna(subset=["maturity_in_months"])
        # Imputation of variables
        for variable in VARIABLES_WITH_MEAN_VALUES_IMPUTATION:
            X[variable + "_imputed"] = X[variable].fillna(np.mean(X[variable]))
        for variable in VARIABLES_WITH_ZERO_VALUE_IMPUTATION:
            X[variable + "_imputed"] = X[variable].fillna(0)
        for variable in VARIABLES_WITH_MOST_FREQUENT_VALUE_IMPUTATION:
            X[variable + "_imputed"] = X[variable].fillna(X[variable].value_counts().idxmax())
        # We create a modality "OTHER" for variable domains with rare events.
        # OTHER category represents 5% of the training contracts.
        X["email_domain_imputed"] = np.where(
            (X["email_domain_imputed"].isin(MOST_FREQUENT_EMAIL_DOMAINS)), X["email_domain_imputed"], "OTHER"
        )
        X = (
            X.assign(
                #####################################################
                # budget dynamical variables and spending behaviour #
                #####################################################
                # the borrowed amount was floored with 6000 value, because of historical evolution of the borrowed amount
                # values opened and then closed.
                # borrowed_amount_imputed=lambda x: x["borrowed_amount"].apply(lambda value: max(value, 6000)),
                housing_expenses=lambda x: x["mortgage_amount"].fillna(0) + x["rent_amount"].fillna(0),
                # this the main income reweighted according the number of months paid during the year (up to 15
                # in PT). The definition was provided by CSM.
                main_income_with_month=np.select(
                    [
                        X["number_of_salaries_per_year"] == 13,
                        X["number_of_salaries_per_year"] == 14,
                        X["number_of_salaries_per_year"] == 15,
                    ],
                    [
                        13 * X["main_net_monthly_income"] / 12,
                        14 * X["main_net_monthly_income"] / 12,
                        15 * X["main_net_monthly_income"] / 12,
                    ],
                    default=X["main_net_monthly_income"],
                ),
                # !!! Attention !!! the fillna here below are normally useless, because they were carried out
                # in the operation above. I kept them for historical reasons.
                # total_expense definition as given by CSM
                # total_expense=lambda x: x["rent_amount"].fillna(0)
                #                         + x["mortgage_amount"].fillna(0)
                #                         + x["ongoing_credits_amount"].fillna(0)
                #                         + (X["borrowed_amount"] / X["maturity_in_months"]).fillna(0),
                # other definition of expenses that we decided to test
                # expenses=X["rent_amount"].fillna(0)
                #          + X["mortgage_amount"].fillna(0)
                #          + X["ongoing_credits_amount"].fillna(0)
                #          + (X["borrowed_amount"] / X["maturity_in_months"]),
                # total_credit_amount=X["ongoing_credits_amount"].fillna(0)
                #                     + (X["borrowed_amount"] / X["maturity_in_months"]),
                total_credit_amount=X["ongoing_credits_amount"].fillna(0) + X["mortgage_amount"].fillna(0),
                budget=X["main_net_monthly_income"].fillna(0)
                - (X["rent_amount"].fillna(0) + X["mortgage_amount"].fillna(0) + X["ongoing_credits_amount"].fillna(0)),
                # other definition of indetebdness_ratio tested - this definition is not used in the final variable choice
                # because we decided not to use borrowed_amount and maturity_in_months variables
                # indetebdness_ratio_new=(
                #                                (X["borrowed_amount"] / X["maturity_in_months"])
                #                                + (X["rent_amount"].fillna(0) + X["mortgage_amount"].fillna(0) + X[
                #                            "ongoing_credits_amount"].fillna(0))
                #                        )
                #                        / X["main_net_monthly_income"],
                # The new assign is here used to leverage the main_income_with_month variable computed in the
                # previous assign statement
            )
            .assign(
                variable_income=lambda x: x["main_income_with_month"] - x["main_net_monthly_income"],
                budget_credits=lambda x: x["main_income_with_month"].fillna(0) - x["total_credit_amount"].fillna(0),
                indetebdness_ratio_credits=lambda x: x["total_credit_amount"].fillna(0)
                / x["main_income_with_month"].fillna(0),
                # the following variables were computed to carry out some data quality tests. In particular, we
                # checked the coherence between the housing status code and the rent and mortgage amount values.
                has_mortgage_filled=lambda x: np.select(
                    [
                        x["mortgage_amount"].isna(),
                    ],
                    [
                        False,
                    ],
                    default=True,
                ),
                has_rent_filled=lambda x: np.select(
                    [
                        x["rent_amount"].isna(),
                    ],
                    [
                        False,
                    ],
                    default=True,
                ),
                tenant_with_rent=lambda x: np.select(
                    [
                        (x["housing_code"] == "TENANT") & (x["rent_amount"].fillna(0) > 0.0),
                    ],
                    [
                        True,
                    ],
                    default=False,
                ),
                landlord_with_mortgage=lambda x: np.select(
                    [
                        (x["housing_code"] == "HOME_OWNERSHIP_WITH_MORTGAGE") & (x["mortgage_amount"].fillna(0) > 0.0),
                    ],
                    [
                        True,
                    ],
                    default=False,
                ),
                landlord_with_mortgage2=lambda x: np.select(
                    [
                        (x["housing_code"] == "HOME_OWNERSHIP_WITHOUT_MORTGAGE")
                        & (x["mortgage_amount"].fillna(0) > 0.0),
                    ],
                    [
                        True,
                    ],
                    default=False,
                ),
            )
            .assign(
                has_variable_income=lambda x: np.select(
                    [
                        x["variable_income"] > 0.0,
                    ],
                    [
                        True,
                    ],
                    default=False,
                ),
            )
        )
        # Return the transformed DataFrame
        return X


class Featuresv2Transformer(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """
    This class leverages the features created in FlattenTransformer. It creates some were present in model 2.1 e.g.,
     personal_age_grp
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def adj_net_monthly_income_creation(x):
            return x["main_net_monthly_income"] * x["number_of_salaries_per_year"] / 12

        return X.assign(
            # previous debt expenses ratio used
            debt_expenses_ratio=lambda x: (
                x["ongoing_credits_amount"] / (x["housing_expenses"] + x["ongoing_credits_amount"])
            ).fillna(-1),
            debt_expenses_ratio_grp=lambda x: np.select(
                [
                    x["debt_expenses_ratio"] == -1,
                    x["debt_expenses_ratio"] >= 0.7,
                    True,
                ],
                ["no_expenses", "above_70_percent", "under_70_percent"],
            ),
            adj_net_monthly_income=X.apply(adj_net_monthly_income_creation, axis=1),
            income_grp=lambda x: pd.cut(
                x["adj_net_monthly_income"],
                bins=[-float("inf"), 925, 1225, 1640, float("inf")],
                labels=["[-Inf,925]", "(925,1225]", "(1225,1640]", "(1640,Inf]"],
                include_lowest=True,
            ),
            # following variables are buckets sometimes used in the previous model versions
            housing_expenses_grp=lambda x: pd.cut(
                x["housing_expenses"],
                bins=[-float("inf"), 0, 110, 300, float("inf")],
                labels=["[-Inf,0]", "(0,110]", "(110,300]", "(300,Inf]"],
                include_lowest=True,
            ),
            income_x_housing_grp=lambda x: np.select(
                [
                    x["income_grp"] == "(1640,Inf]",
                    (x["income_grp"] == "(1225,1640]") & (x["housing_expenses_grp"].isin(["(110,300]", "(300,Inf]"])),
                    (x["income_grp"] == "(1225,1640]") & (x["housing_expenses_grp"] == "(0,110]"),
                    (x["income_grp"] == "(925,1225]")
                    & (x["housing_expenses_grp"].isin(["(0,110]", "(110,300]", "(300,Inf]"])),
                    True,
                ],
                ["Group_1", "Group_2", "Group_3", "Group_3", "Group_4"],
            ),
            sector_code=lambda x: np.where(x["profession_code"] == "RETIREMENT", "RETIREE", x["sector_code"]),
            professional_age_tmp=lambda x: np.where(
                (x["professional_age"].isna()) | (12 * x["professional_age"] < 6) | (x["sector_code"] == "RETIREE"),
                np.nan,
                12 * x["professional_age"],
            ),  # TODO: adapt to concord with current score
            professional_age_grp=lambda x: pd.cut(
                x["professional_age_tmp"],
                bins=[-float("inf"), 38, 120, 230, float("inf")],
                labels=["[-Inf,38]", "(38,120]", "(120,230]", "(230,Inf]"],
                include_lowest=True,
            ),
            prof_status_grp=lambda x: np.select(
                [
                    (x["sector_code"] == "PRIVATE_SECTOR")
                    & (x["professional_age_grp"].isin(["(120,230]", "(230,Inf]"])),
                    x["sector_code"] == "RETIREE",
                    (x["sector_code"] == "PUBLIC_SECTOR")
                    & (x["professional_age_grp"].isin(["(120,230]", "(230,Inf]"])),
                    True,
                ],
                ["Group_1", "Group_2", "Group_2", "Group_3"],
            ),
            personal_age_grp=lambda x: pd.cut(
                x["personal_age"],
                bins=[-float("inf"), 37, 43, 52, float("inf")],
                labels=["[-Inf,37]", "(37,43]", "(43,52]", "(52,Inf]"],
                include_lowest=True,
            ),
            housing_code_grp=lambda x: np.where(
                x["housing_code"].isin(["THIRD_PARTY_PROVIDED_LODGING", "EMPLOYER_PROVIDED_LODGING"]),
                "THIRD_PARTY",
                x["housing_code"],
            ),
            housing_status_x_age_grp=lambda x: np.select(
                [
                    x["housing_code_grp"] == "HOME_OWNERSHIP_WITH_MORTGAGE",
                    (x["housing_code_grp"] == "HOME_OWNERSHIP_WITHOUT_MORTGAGE")
                    & (x["personal_age_grp"].isin(["(52,Inf]"])),
                    (x["housing_code_grp"] == "HOME_OWNERSHIP_WITHOUT_MORTGAGE")
                    & (x["personal_age_grp"].isin(["(43,52]", "(37,43]"])),
                    (x["housing_code_grp"] == "TENANT")
                    & (x["personal_age_grp"].isin(["(52,Inf]", "(43,52]", "(37,43]", "[-Inf,37]"])),
                    True,
                ],
                ["Group_1", "Group_1", "Group_2", "Group_2", "Group_3"],
            ),
            marital_status_grp=lambda x: np.where(
                x["marital_status_code"].isin(["COHABITING", "MARRIED"]),
                "IN_A_RELATIONSHIP",
                "SINGLE",
            ),
        )


class OneHotEncoderV2_1(sklearn.preprocessing.OneHotEncoder):
    """
    The purpose of this class to create the features as they exist in the current model in production (i.e. model 2.1)
    """

    def __init__(self):
        super().__init__(sparse_output=False, handle_unknown="ignore", drop="first")

    def fit(self, *args, **kwargs):
        self.__dict__.update(
            {
                "_infrequent_enabled": False,
                "n_features_in_": 5,
                "feature_names_in_": np.array(
                    FEATURES_TO_ENCODE_2_1,
                    dtype=object,
                ),
                "categories_": [
                    np.array(
                        ["above_70_percent", "no_expenses", "under_70_percent"],
                        dtype=object,
                    ),
                    np.array(["Group_1", "Group_2", "Group_3", "Group_4"], dtype=object),
                    np.array(["Group_1", "Group_2", "Group_3"], dtype=object),
                    np.array(["Group_1", "Group_2", "Group_3"], dtype=object),
                    np.array(["IN_A_RELATIONSHIP", "SINGLE"], dtype=object),
                ],
                "drop_idx_": np.array([0, 0, 0, 0, 0], dtype=object),
                "_drop_idx_after_grouping": np.array([0, 0, 0, 0, 0], dtype=object),
                "_n_features_outs": [2, 3, 2, 2, 1],
            }
        )
        return self


class LogisticRegressionV2_1(sklearn.linear_model.LogisticRegression):
    """
    The purpose of this class is to recreate the exact same Logistic Regression currently used in the model V2 in
    production. In order to do that, the hyperparameters and coefficients are fixed with the exact same values.
    """

    def __init__(self):
        kwargs = HYPERPARAMETERS_2_1
        super().__init__(**kwargs)

    def fit(self, *args, **kwargs):
        self.__dict__.update(
            {
                "feature_names_in_": np.array(
                    FEATURES_2_1,
                    dtype=object,
                ),
                "n_features_in_": 10,
                "classes_": np.array([0.0, 1.0]),
                "n_iter_": np.array([14], dtype=np.int32),
                "coef_": np.array([COEFFICIENTS_2_1]),
                "intercept_": np.array([-3.4966]),
            }
        )
        return self


class OptBinning2DEncoderV2(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """
    This class leverages the 2D Optimal binning to recreate the same interactions present in the model in production
    (i.e. v2), but using the Optimal Binning method instead of the buckets currently in use.
    """

    def __init__(self):
        self.optb_debt_expenses_ratio_grp = optbinning.OptimalBinning(name="debt_expenses_ratio")
        self.optb_income_x_housing_grp = optbinning.OptimalBinning2D(
            name_x="adj_net_monthly_income", name_y="housing_expenses"
        )
        self.optb_prof_status_grp = optbinning.OptimalBinning2D(
            name_x="profession_code", name_y="professional_age", dtype_x="categorical"
        )
        self.optb_housing_age = optbinning.OptimalBinning2D(
            name_x="housing_code", name_y="personal_age", dtype_x="categorical"
        )
        self.optb_marital_status = optbinning.OptimalBinning(dtype="categorical")

    def fit(self, X, y=None):
        self.optb_debt_expenses_ratio_grp.fit(X["debt_expenses_ratio"], y)
        self.optb_income_x_housing_grp.fit(X["adj_net_monthly_income"], X["housing_expenses"], y)
        self.optb_prof_status_grp.fit(X["profession_code"], X["professional_age"], y)
        self.optb_housing_age.fit(X["housing_code"], X["personal_age"], y)
        self.optb_marital_status.fit(X["marital_status_code"], y)
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(
            data={
                "debt_expenses_ratio_grp": self.optb_debt_expenses_ratio_grp.transform(X["debt_expenses_ratio"]),
                "income_x_housing_grp": self.optb_income_x_housing_grp.transform(
                    X["adj_net_monthly_income"], X["housing_expenses"]
                ),
                "prof_status_grp": self.optb_prof_status_grp.transform(X["profession_code"], X["professional_age"]),
                "housing_status_x_age_grp": self.optb_housing_age.transform(X["housing_code"], X["personal_age"]),
                "marital_status_grp": self.optb_marital_status.transform(X["marital_status_code"]),
            },
            index=X.index,
        )
        return X


def json_normalize(X):
    return pd.json_normalize(X["payload"])


# Pipeline for the preprocessing of the features needed in new model candidate 3.1
# # pipeline is called in the preprocess-stage
# preprocess_pipeline_model_3p1 = Pipeline(
#     steps=[
#         (
#             "json_normalize",
#             sklearn.preprocessing.FunctionTransformer(json_normalize),
#         ),
#         ("slicer", FlattenTransformer()),
#     ],
# )

# Pipeline to reconstruct model 2.1 currently in production
preprocess_pipeline_model_2p1 = Pipeline(
    steps=[
        (
            "json_normalize",
            sklearn.preprocessing.FunctionTransformer(json_normalize),
        ),
        ("slicer", FlattenTransformer()),
        ("v2", Featuresv2Transformer()),
    ],
)

# declaration of the features as in the model in production v2.
v2_features = [
    "debt_expenses_ratio_grp",
    "income_x_housing_grp",
    "prof_status_grp",
    "housing_status_x_age_grp",
    "marital_status_grp",
]

# # declaration of the features tested for the new candidate model v3.
# v3_features = [
#     # Declared variables
#     "personal_age_imputed",
#     "bank_age_imputed",
#     "bank_code",
#     # "mortgage_amount",
#     "housing_code",
#     "marital_status_code",
#     # "rent_amount",
#     "professional_age_imputed",
#     "profession_code",
#     # "debt_expenses_ratio",
#     "professional_situation_code",
#     "phone_prefix_imputed",
#     "postal_region",
#     # "project_type_code",
#     # # Correlations
#     # "correlation_target",
#     # "business_provider_code",
#     "email_domain_imputed",
#     # impute missing with most frequent and set emails domain lowercase -> perhaps create a categpry that is not hotmail an gmail
#     # "gender_code",
#     # "nationality_code",
#     # "is_repeat_business",
#     # "maturity",
#     "borrowed_amount_imputed",
#     "maturity_in_months",
#     # "installment",
#     # "budget",
#     # "tim_in_hours",
#     # "has_coborrower",
#     # differences for numerical vars
#     # "mortgage_amount_difference",
#     # "rent_amount_difference",
#     "professional_age_difference_imputed",
#     # "personal_age_difference_imputed",
#     # "bank_age_difference_imputed",
#     # Nicolas' feature
#     # "partner_income",
#     # "total_income",
#     # "total_expense",
#     # "budget_csm",
#     # "paid_month_code_imputed",
#     # more correlations:
#     # "freq_previously_funded_applications_per_days",
#     # "number_of_applications_before_funding",
#     # "number_of_previously_funded_applications",
#     # "corr__total_incidents",
#     # "corr__largest_incident"
#     # new vars
#     # "total_credit_amount",
#     # "budget_credits",
#     # "indetebdness_ratio_credits",
#     # new vars
#     # "has_mortgage_filled",
#     # "has_rent_filled",
#     # "tenant_with_rent",
#     # "landlord_with_mortgage",
#     # "landlord_with_mortgage2",
#     "ongoing_credits_amount_imputed",
#     # "ongoing_credits_car",
#     # "ongoing_credits_realestate",
#     "ongoing_credits_amount_difference",
#     "rent_amount_imputed",
#     "mortgage_amount_imputed",
#     "indetebdness_ratio_new",
#     # "main_income_with_month",
#     "main_net_monthly_income",
#     "variable_income",
#     "main_net_monthly_income_difference",
#     # "expenses",
# ]
#
# # declaration of the features tested for the new candidate model v3.
# v3_features = [
#     "personal_age_imputed",
#     "bank_age_imputed",
#     "bank_code",
#     "housing_code",
#     "marital_status_code",
#     "professional_age_imputed",
#     "profession_code",
#     "professional_situation_code",
#     "phone_prefix_imputed",
#     "postal_region",
#     "email_domain_imputed",
#     # "borrowed_amount_imputed",
#     # "maturity_in_months",
#     "professional_age_difference_imputed",
#     "ongoing_credits_amount_imputed",
#     "ongoing_credits_amount_difference",
#     "rent_amount_imputed",
#     "mortgage_amount_imputed",
#     "indetebdness_ratio_credits",
#     "main_net_monthly_income",
#     "variable_income",
#     "main_net_monthly_income_difference",
#     "has_variable_income",
# ]
#
# # declaration of the features present in v3_features, but that should be of categorical dtype
# v3_cat_features = [
#     "bank_code",
#     "housing_code",
#     # "paid_month_code_imputed",
#     "phone_prefix_imputed",
#     # "project_type_code",
#     # "ongoing_credits_realestate",
#     "marital_status_code",
#     "profession_code",
#     "postal_region",
#     "professional_situation_code",
#     # "is_repeat_business",
#     # "business_provider_code",
#     "email_domain_imputed",
#     "has_variable_income",
#     # "gender_code",
#     # "nationality_code",
#     # "has_coborrower",
#     # "has_mortgage_filled",
#     # "has_rent_filled",
#     # "ongoing_credits_car",
#     # "tenant_with_rent",
#     # "landlord_with_mortgage",
#     # "landlord_with_mortgage2",
# ]
