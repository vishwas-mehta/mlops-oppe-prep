from feast import Entity, FeatureView, Field, FileSource
from feast.value_type import ValueType
from feast.types import Float64, Int64
from datetime import timedelta

stock = Entity(
    name="stock",
    join_keys=["stock"],
    value_type=ValueType.STRING,
    description="NSE stock ticker symbol"
)

processed_data_source = FileSource(
    path="../data/processed/v0_v1/combined_processed_data.parquet",
    timestamp_field="timestamp",
)

stock_features_fv = FeatureView(
    name="stock_technical_features",
    entities=[stock],
    ttl=timedelta(days=7),
    schema=[
        Field(name="rolling_avg_10", dtype=Float64),
        Field(name="volume_sum_10", dtype=Float64),
        Field(name="target", dtype=Int64),
    ],
    source=processed_data_source,
)
