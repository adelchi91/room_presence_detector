import json
import os
import pathlib
import pickle

import pandas as pd
from yc_datamarty.utils import validate_payload_from_pydantic_model

from payload_validation.pydantic_model import FinalModel as PydanticModel

""""
This dvc stage step is used to validate the data retrieved from the datamart and validate it according to the pydantic
model.
Any record that would not be correctly validate would be shown in the following dataframe
validated_dataset[lambda x: ~x.validation_error.isna()]["validation_error"]
The validated jscon schema is retrieved with PydanticModel.schema_json() and can then be communicated to the score
team if no error is encountered.
"""

data = pathlib.Path("data")

validated_dataset_path = data / "validate_payloads/validated_payloads.pkl"

X = pd.read_pickle(data / "payloads_dataset.pkl")

validated_dataset = X.assign(
    validation_results=lambda x: x["payload"].apply(
        lambda y: validate_payload_from_pydantic_model(payload=y, pydantic_model=PydanticModel)
    ),
    validated_payload=lambda x: x["validation_results"].apply(lambda y: y[0]),
    validation_status=lambda x: x["validation_results"].apply(lambda y: y[1]),
    validation_error=lambda x: x["validation_results"].apply(lambda y: y[2]),
).drop(columns="validation_results")

print(validated_dataset[lambda x: ~x.validation_error.isna()]["validation_error"].iloc[0])
# try:
#     assert all(validated_dataset["validation_status"])
# except Exception:
#     raise ValueError(
#         "Validation failed for some payloads, please check your Pydantic model !"
#         "\n-> For more information, please inspect the dataframe 'validated_dataset' in debug mode."
#     )

pathlib.Path.mkdir(pathlib.Path("data", "validate_payloads"), exist_ok=True)

# Save pickle files
with open(pathlib.Path(validated_dataset_path), "wb") as f:
    pickle.dump(
        validated_dataset[["request_id", "validated_payload"]].rename(columns={"validated_payload": "payload"}), f
    )


# saving json payload to send to score
file_path = os.path.join("src", "payload_validation", "payload_for_score_team.json")
# save
data = PydanticModel.schema_json()
with open("src/payload_validation/payload_for_score_team.json", "w") as file:
    json.dump(data, file)
