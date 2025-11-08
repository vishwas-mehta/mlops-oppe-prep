import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def train_and_evaluate_model(processed_data_path: str, model_output_path: str):
    """
    Loads processed data, trains a Logistic Regression model, evaluates it,
    and saves the trained model.
    """
    print(f"Loading data from {processed_data_path}...")
    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_data_path}")
        return

    features = ['rolling_avg_10', 'volume_sum_10']
    target = 'target'

    if not all(col in df.columns for col in features + [target]):
        print("Error: DataFrame is missing required feature or target columns.")
        return

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    # Train v0 model
    train_and_evaluate_model(
        'data/processed/v0/combined_processed_data.csv',
        'models/v0/model.joblib'
    )
    
    # Train v0_v1 model
    train_and_evaluate_model(
        'data/processed/v0_v1/combined_processed_data.csv',
        'models/v0_v1/model.joblib'
    )
