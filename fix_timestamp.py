import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Read the parquet file
df = pd.read_parquet('data/processed/v0_v1/combined_processed_data.parquet')

# Convert timestamp column to proper datetime with timezone
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

# Save back to parquet
df.to_parquet('data/processed/v0_v1/combined_processed_data.parquet')

print("✅ Fixed timestamp column!")
print(f"✅ Timestamp dtype is now: {df['timestamp'].dtype}")
print(f"✅ Sample timestamps: {df['timestamp'].head()}")
