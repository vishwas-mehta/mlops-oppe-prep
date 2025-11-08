import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")

def train_and_evaluate_model_with_mlflow(processed_data_path: str, model_output_path: str, experiment_name: str):
    mlflow.set_experiment(experiment_name)
    
    print(f"Loading data from {processed_data_path}...")
    df = pd.read_parquet(processed_data_path)
    
    features = ['rolling_avg_10', 'volume_sum_10']
    target = 'target'
    
    X = df[features]
    y = df[target]
    
    with mlflow.start_run():
        mlflow.log_param("dataset_path", processed_data_path)
        mlflow.log_param("features", features)
        mlflow.log_param("n_samples", len(X))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("test_split", 0.2)
        mlflow.log_param("random_state", 42)
        
        print("Training Logistic Regression model...")
        model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("max_iter", 1000)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ Model logged to MLflow and saved to {model_output_path}")
        print(f"✅ MLflow Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Training Model v0")
    print("="*60)
    train_and_evaluate_model_with_mlflow(
        "data/processed/v0/combined_processed_data.parquet",
        "models/v0/model.joblib",
        "stock_prediction_v0"
    )
    
    print("\n" + "="*60)
    print("Training Model v0_v1")
    print("="*60)
    train_and_evaluate_model_with_mlflow(
        "data/processed/v0_v1/combined_processed_data.parquet",
        "models/v0_v1/model.joblib",
        "stock_prediction_v0_v1"
    )
