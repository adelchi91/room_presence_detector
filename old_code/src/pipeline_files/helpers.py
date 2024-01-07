import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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
VARIABLES_WITH_MOST_FREQUENT_VALUE_IMPUTATION = ["email_domain", "phone_prefix", "postal_region"]

MOST_FREQUENT_EMAIL_DOMAINS = ["gmail.com", "hotmail.com", "sapo.pt", "live.com.pt", "outlook.pt", "outlook.com"]


# Custom transformer to modify "email_domain_imputed"
class ModifyEmailDomainTransformer(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """
    This class is a very simple transform method used in the pipeline in order to create an expert bucket
    of email addresses. This is done because we do not trust how optimal binning splits this variable in different
    buckets.
    """

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def set_output(self, *, transform=None):
        self._is_set_output = True
        return self

    def transform(self, X, y=None):
        X["email_domain_imputed"] = np.where(
            (X["simpleimputer-3__email_domain"].isin(MOST_FREQUENT_EMAIL_DOMAINS)),
            X["simpleimputer-3__email_domain"],
            "OTHER",
        )
        return X


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

    @staticmethod
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
        if filtered_data.isnull().all():
            flattened_df["monthly_amount"] = None
        else:
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

    @staticmethod
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

    @staticmethod
    def age_in_year(beg, end):
        return (end - beg) / np.timedelta64(1, "Y")

    @staticmethod
    def coalesce_situation(path, X):
        return X[f"verified_situation.{path}"].combine_first(X[f"declared_situation.{path}"])

    @staticmethod
    def credit_lines_dataframe_expansion(df, data_type="verified_situation"):
        # select variables of interest
        df = df[[f"{data_type}.expenses.ongoing_credits.credit_lines", "request_id"]].copy()
        # Exploding credit lines: creating a new dataframe with the exploded values
        expanded_df = df.explode(f"{data_type}.expenses.ongoing_credits.credit_lines")
        # Reset the index to ensure consistent merging later
        expanded_df = expanded_df.reset_index(drop=True)
        return expanded_df

    def set_output(self, *, transform=None):
        self._is_set_output = True
        return self

    def transform(self, X_input, y=None):
        # saving a copy, that will be used later
        # The transform method of a transformer should not modify the input data in place. Instead, it should return a
        # new DataFrame or array with the transformed features. Since your custom transformer modifies X in place, this
        # can lead to inconsistencies between different parts of your pipeline and cross-validation process.
        X = X_input.copy()
        X_temp = X_input.copy()
        # Exploding credit lines
        expanded_df = self.credit_lines_dataframe_expansion(X, data_type="verified_situation")
        # ongoing_credits_amount
        X = self.create_ongoing_credits_amount(X, expanded_df, data_type="verified_situation")
        # specific credits
        X = self.create_ongoing_credits_specific(X, expanded_df, specific="CARLOAN")
        X = self.create_ongoing_credits_specific(X, expanded_df, specific="PERSONALLOAN")
        X = self.create_ongoing_credits_specific(X, expanded_df, specific="OTHER")
        X = self.create_ongoing_credits_specific(X, expanded_df, specific="LIQUIDITY")
        X = self.create_ongoing_credits_specific(X, expanded_df, specific="REALESTATELOAN")
        X = self.create_ongoing_credits_specific(X, expanded_df, specific="DEBTS")
        # differences in credit declarations/verifications
        # Exploding credit lines - declared data
        expanded_df_declared = self.credit_lines_dataframe_expansion(X_temp, data_type="declared_situation")
        # ongoing_credits_amount - declared data
        X_temp = self.create_ongoing_credits_amount(X_temp, expanded_df_declared, data_type="declared_situation")
        X = X.merge(X_temp[["request_id", "ongoing_credits_amount_declared"]], on="request_id", how="left")
        X = X.assign(
            ongoing_credits_amount_difference=lambda x: x["ongoing_credits_amount_declared"].fillna(0)
            - x["ongoing_credits_amount"].fillna(0)
        )

        # other variables of interest
        application_date = pd.to_datetime(X["application_date"], utc=True)
        date_of_birth = pd.to_datetime(X["declared_situation.applicant.personal.date_of_birth"], utc=True)
        bank_account_opening_date = pd.to_datetime(
            self.coalesce_situation("banking.bank_account_opening_date", X), utc=True
        )
        professional_situation_start_date = pd.to_datetime(
            self.coalesce_situation("applicant.employment_situation.professional_situation_start_date", X),
            utc=True,
        )
        # The coalesce method is leveraged in order to consider the verified variable first, whenever present, or
        # the declared info otherwise.
        for input_path, output_name in VARIABLES_TO_COALESCE.items():
            X[output_name] = self.coalesce_situation(input_path, X)
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
            marital_status_code=self.coalesce_situation("marital_situation.marital_status_code", X),
            #############################
            # age computations in years #
            #############################
            personal_age=self.age_in_year(date_of_birth, application_date),
            bank_age=self.age_in_year(bank_account_opening_date, application_date),
            professional_age=self.age_in_year(professional_situation_start_date, application_date),
            ##############################################################
            # filling missing values and renaming of specific modalities #
            ##############################################################
            professional_situation_code=X["professional_situation_code"].replace({"PENSION": "PENSIONER_RETIRED"}),
            number_of_salaries_per_year=X["number_of_salaries_per_year"].fillna(12),
            ###############
            # differences #
            ###############
            personal_age_difference=self.age_in_year(
                pd.to_datetime(X["declared_situation.applicant.personal.date_of_birth"], utc=True), application_date
            )
            - self.age_in_year(
                pd.to_datetime(X["verified_situation.applicant.personal.date_of_birth"], utc=True), application_date
            ),
            bank_age_difference=self.age_in_year(
                pd.to_datetime(X["declared_situation.banking.bank_account_opening_date"], utc=True), application_date
            )
            - self.age_in_year(bank_account_opening_date, application_date),
            mortgage_amount_difference=X["declared_situation.expenses.mortgage_amount"].fillna(0)
            - X["mortgage_amount"].fillna(0),
            main_net_monthly_income_difference=X["declared_situation.incomes.main_net_monthly_income"].fillna(0)
            - X["main_net_monthly_income"],
            professional_age_difference=self.age_in_year(
                pd.to_datetime(
                    X["declared_situation.applicant.employment_situation.professional_situation_start_date"], utc=True
                ),
                application_date,
            )
            - self.age_in_year(professional_situation_start_date, application_date),
            rent_amount_difference=X["declared_situation.expenses.rent_amount"]
            - self.coalesce_situation("expenses.rent_amount", X),
            #################
            # New variables #
            #################
            postal_region=X["postal_code"].str[0:2],
            # we deliberately choose to use the verified cell_phone_number information
            phone_prefix=X["cell_phone_number"].str[0:6],
            # we deliberately choose to use the verified email information
            email_domain=X["personal.email"].str.replace(".*@", "", regex=True).str.lower(),
            ongoing_credits_count=X["verified_situation.expenses.ongoing_credits.credit_lines"].apply(
                lambda credits: len(credits)
            ),
        )
        X = (
            X.assign(
                #####################################################
                # budget dynamical variables and spending behaviour #
                #####################################################
                # the borrowed amount was floored with 6000 value, because of historical evolution of the borrowed
                # amount values opened and then closed.
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
                total_credit_amount=X["ongoing_credits_amount"].fillna(0) + X["mortgage_amount"].fillna(0),
                budget=X["main_net_monthly_income"].fillna(0)
                - (X["rent_amount"].fillna(0) + X["mortgage_amount"].fillna(0) + X["ongoing_credits_amount"].fillna(0)),
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


def json_normalize(X):
    return pd.json_normalize(X["payload"])


# Pipeline for the preprocessing of the features needed in new model candidate 3.1
# pipeline is called in the preprocess-stage
preprocess_pipeline_model_3p1 = Pipeline(
    steps=[
        (
            "json_normalize",
            sklearn.preprocessing.FunctionTransformer(json_normalize),
        ),
        ("slicer", FlattenTransformer()),
        (
            "imputation",
            make_column_transformer(
                (
                    SimpleImputer(strategy="mean", missing_values=np.nan, fill_value=None, verbose=0, copy=True),
                    VARIABLES_WITH_MEAN_VALUES_IMPUTATION,
                ),
                (
                    SimpleImputer(strategy="constant", missing_values=np.nan, fill_value=0, verbose=0, copy=True),
                    VARIABLES_WITH_ZERO_VALUE_IMPUTATION,
                ),
                (
                    SimpleImputer(
                        strategy="most_frequent", missing_values=np.nan, fill_value=None, verbose=0, copy=True
                    ),
                    VARIABLES_WITH_MOST_FREQUENT_VALUE_IMPUTATION,
                ),
                remainder="passthrough",
            ),
        ),
        # Modify "email_domain_imputed"
        ("modify_email_domain", ModifyEmailDomainTransformer()),
    ],
).set_output(transform="pandas")

# declaration of the features tested for the new candidate model v3.
v3_features = [
    "simpleimputer-1__personal_age",
    "simpleimputer-1__bank_age",
    "remainder__bank_code",
    "remainder__housing_code",
    "remainder__marital_status_code",
    "simpleimputer-1__professional_age",
    "remainder__profession_code",
    "remainder__professional_situation_code",
    "simpleimputer-3__phone_prefix",
    "simpleimputer-3__postal_region",
    "email_domain_imputed",
    # "borrowed_amount_imputed",
    # "maturity_in_months",
    "simpleimputer-2__professional_age_difference",
    "simpleimputer-2__ongoing_credits_amount",
    "remainder__ongoing_credits_amount_difference",
    "simpleimputer-2__rent_amount",
    "simpleimputer-2__mortgage_amount",
    "remainder__indetebdness_ratio_credits",
    "remainder__main_net_monthly_income",
    "remainder__variable_income",
    "remainder__main_net_monthly_income_difference",
    "remainder__has_variable_income",
]

# declaration of the features present in v3_features, but that should be of categorical dtype
v3_cat_features = [
    "remainder__bank_code",
    "remainder__housing_code",
    "simpleimputer-3__phone_prefix",
    "remainder__marital_status_code",
    "remainder__profession_code",
    "simpleimputer-3__postal_region",
    "remainder__professional_situation_code",
    "email_domain_imputed",
    "remainder__has_variable_income",
]
