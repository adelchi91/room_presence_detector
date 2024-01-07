import joblib
import pandas as pd

from src.pipeline_files.helpers import Featuresv2Transformer

clf_v2_1 = joblib.load("data/models/v2.1.joblib")
df = pd.read_csv(
    "data/Novobanco_score_sent_DataS_V2_csv.csv",
    delimiter=";",
    parse_dates=["Application_date"],
)
X0 = df.rename(
    columns={
        "Customer_age_at_application": "personal_age",
        "Ongoing_credits_instalment_amount": "ongoing_credits_amount",
        "monthly_main_net_income": "main_net_monthly_income",
        "Nb_paid_month": "number_of_paid_months_per_year",
        "Type of contract": "sector_code",
        "Job_age_months": "professional_age",
        "Housing_status.1": "housing_code",
        "Marital_status.1": "marital_status_code",
        "Housing_expenses": "housing_expenses",
    }
).assign(
    profession_code=lambda X: "OTHER",
    professional_age=lambda X: X["professional_age"] / 12,
    mortgage_amount=lambda X: X.apply(
        lambda row: row["housing_expenses"] if row["housing_code"] == "HOME_OWNERSHIP_WITH_MORTGAGE" else 0,
        axis=1,
    ),
    rent_amount=lambda X: X["housing_expenses"] - X["mortgage_amount"],
    ongoing_credits_amount=lambda X: X["ongoing_credits_amount"].str.replace(" ", "").astype(float),
    main_net_monthly_income=lambda X: X["main_net_monthly_income"].str.replace(" ", "").astype(float),
)
df["yc_proba"] = clf_v2_1.predict_proba(Featuresv2Transformer().transform(X0))[:, 1]
df["yc_score"] = (10000 * (1 - df["yc_proba"])).astype(int)
df.to_csv("data/Novobanco_score_sent_DataS_V3_csv.csv", index=False, sep=";")
