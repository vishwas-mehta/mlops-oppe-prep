import pandas as pd

# Read the CSV file
df = pd.read_csv('data/processed/v0_v1/combined_processed_data.csv')

# Save as Parquet file
df.to_parquet('data/processed/v0_v1/combined_processed_data.parquet')

print("✅ Successfully converted CSV to Parquet!")
print(f"✅ Created: data/processed/v0_v1/combined_processed_data.parquet")
