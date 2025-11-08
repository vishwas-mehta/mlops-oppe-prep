import joblib
import pandas as pd
import os

def test_model_loading_and_prediction():
    """Test if the model can be loaded and makes predictions without error."""
    # Use v0 model path for the test
    model_path = "models/v0/model.joblib"
    
    # This assertion will fail if the file doesn't exist.
    # In a CI/CD pipeline, this file will be pulled by DVC.
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

    model = joblib.load(model_path)
    
    # Create dummy input data matching the feature names
    dummy_data = pd.DataFrame({
        'rolling_avg_10': [100.0, 105.0],
        'volume_sum_10': [1000.0, 1200.0]
    })
    
    predictions = model.predict(dummy_data)
    
    assert predictions is not None, "Model prediction failed"
    assert len(predictions) == len(dummy_data), "Prediction output length mismatch"
    assert all(p in [0, 1] for p in predictions), "Predictions are not binary (0 or 1)"
    print("test_model_loading_and_prediction PASSED")

if __name__ == "__main__":
    # Create a dummy model file for local testing if it doesn't exist
    if not os.path.exists("models/v0/model.joblib"):
        from sklearn.linear_model import LogisticRegression
        os.makedirs("models/v0", exist_ok=True)
        dummy_model = LogisticRegression()
        dummy_model.fit([[0]], [0])
        joblib.dump(dummy_model, "models/v0/model.joblib")
        
    test_model_loading_and_prediction()
