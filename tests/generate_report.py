import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import os
import sys

# Add src to the Python path so we can import features.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def generate_markdown_report(model_path, data_path, output_report_path="report.md"):
    """
    Generates a Markdown report with model metrics.
    """
    print(f"Generating report for model at {model_path} using data from {data_path}...")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    features = ['rolling_avg_10', 'volume_sum_10']
    target = 'target'

    df_clean = df.dropna(subset=features + [target])
    
    X_test = df_clean[features]
    y_test = df_clean[target]

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    with open(output_report_path, 'w') as f:
        f.write(f"# Model Performance Report\n\n")
        f.write(f"**Model Path:** `{model_path}`\n")
        f.write(f"**Data Path:** `{data_path}`\n\n")
        f.write(f"## Overall Accuracy\n")
        f.write(f"`{accuracy:.4f}`\n\n")
        f.write(f"## Classification Report\n")
        f.write(pd.DataFrame(report).transpose().to_markdown())

    print(f"Report successfully generated at {output_report_path}")

if __name__ == "__main__":
    # Generate report for the latest model (v0_v1)
    generate_markdown_report(
        "models/v0_v1/model.joblib",
        "data/processed/v0_v1/combined_processed_data.csv"
    )
