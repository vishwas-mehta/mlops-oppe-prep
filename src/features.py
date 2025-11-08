import pandas as pd
import os

def create_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling_avg_10, volume_sum_10, and the target variable.
    Args:
        df (pd.DataFrame): Input DataFrame with 'timestamp', 'close', 'volume' columns.
    Returns:
        pd.DataFrame: DataFrame with added features and target.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Feature 1: 10-min moving average of close price
    df['rolling_avg_10'] = df['close'].rolling(window=10, closed='right').mean()

    # Feature 2: Total volume traded over 10 min
    df['volume_sum_10'] = df['volume'].rolling(window=10, closed='right').sum()

    # Target variable: Predict if stock will close 5 minutes later at a higher price.
    df['future_close_5min'] = df['close'].shift(-5)
    df['target'] = (df['future_close_5min'] > df['close']).astype(int)

    # Drop rows where features or target cannot be computed (due to NaN values)
    df = df.dropna()
    
    return df

def process_and_combine_data(data_version: str, output_path: str):
    """
    Reads all CSV files from a directory, processes them, and saves a combined CSV.
    """
    data_dir = f'data/{data_version}'
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not all_files:
        print(f"No CSV files found in {data_dir}. Skipping.")
        return

    df_list = []
    for file in all_files:
        print(f"Processing {file}...")
        df = pd.read_csv(file)
        # Extract stock name from filename and add as a column
        df['stock'] = os.path.basename(file).split('__')[0]
        
        # Ensure correct column names if they are different
        df.rename(columns={'DATETIME': 'timestamp', 'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'VOLUME': 'volume'}, inplace=True)

        processed_df = create_features_and_target(df)
        df_list.append(processed_df)

    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined processed data to {output_path}")

if __name__ == "__main__":
    # Process v0 data
    process_and_combine_data('v0', 'data/processed/v0/combined_processed_data.csv')
    
    # Process v0 and v1 data together
    process_and_combine_data('v0', 'data/processed/v0_v1/temp_v0.csv')
    process_and_combine_data('v1', 'data/processed/v0_v1/temp_v1.csv')
    
    df_v0 = pd.read_csv('data/processed/v0_v1/temp_v0.csv')
    df_v1 = pd.read_csv('data/processed/v0_v1/temp_v1.csv')
    combined_v0_v1 = pd.concat([df_v0, df_v1], ignore_index=True)
    combined_v0_v1.to_csv('data/processed/v0_v1/combined_processed_data.csv', index=False)
    
    # Clean up temp files
    os.remove('data/processed/v0_v1/temp_v0.csv')
    os.remove('data/processed/v0_v1/temp_v1.csv')
