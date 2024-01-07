import json
from pathlib import Path

from api_gateway.payload_generation.payload_generation import generate_payload_template
from yc_datamarty.query import QueryHandler
from yc_datamarty.utils import build_payload_query

# YOU NEED TO ADAPT THESE VALUES TO YOUR NEEDS !
MY_CONTEXT = "payload_template"  # "toto"
MY_WORKFLOW = None
MY_BASE_DATASET = "yc-data-science.pt"
MY_BASE_TABLE = "datamart_all_snapshot_pt_2023_08_10"
BANK_READER_ENRICHMENT_OUTPUT_VERSION = None
###

# Instantiate BigQuery client
bq_runner = QueryHandler()

# Load request_ids query
select_rows_query = (Path(__file__).parent.parent / "sql" / "select_rows_query.sql").read_text()

# Load request_ids query
select_targets_query = (Path(__file__).parent.parent / "sql" / "select_targets_query.sql").read_text()

# Load configuration
with open(Path(__file__).parent / "config.json", "r") as f:
    config = json.load(f)


# Build payload template from pydantic model
payload_template_path = generate_payload_template(  # noqa: C901
    context=MY_CONTEXT,
    output_folder=Path(__file__).parent.parent / "payload_validation",
    batch_size=1,
    pydantic_model_path=Path(__file__).parent.parent / "payload_validation" / "pydantic_model.py",
    bank_reader_enrichment_output_version=BANK_READER_ENRICHMENT_OUTPUT_VERSION,
    workflow=MY_WORKFLOW,
)

with open(payload_template_path, "r") as f:
    payload_template = json.load(f)

# Build payloads from datamart
payloads_query = build_payload_query(
    query_handler=bq_runner,
    payload_template=payload_template,
    forced_aliases=config["forced_aliases"],
    forced_defaults=[],  # config["forced_defaults"],
    snapshot_path={"dataset": MY_BASE_DATASET, "table": MY_BASE_TABLE},
    temp_table_dataset="yc-data-science.trash",
    request_ids_query=select_rows_query,
)

X = bq_runner.execute(query=payloads_query)
X["payload"] = X["payload"].apply(lambda x: json.loads(x))
y = bq_runner.execute(query=select_targets_query)

data = Path("data")
data.mkdir(exist_ok=True)

X.to_pickle(data / "payloads_dataset.pkl")
y.to_pickle(data / "targets_dataset.pkl")
