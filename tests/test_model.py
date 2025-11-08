import joblib
import pandas as pd
import numpy as np
import os
import pytest

def test_model_loading_and_prediction():
    """Test if the model can be loaded and makes predictions without error."""
    model_path = "models/v0/model.joblib"
    
    # Skip test in CI if model file is not available (tracked by DVC)
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found at {model_path}. This is expected in CI without DVC pull.")
    
    # Load the model
    model = joblib.load(model_path)
    
    # Create dummy input data matching the expected feature shape
    dummy_data = pd.DataFrame({
        'rolling_avg_10': [100.0, 101.0],
        'volume_sum_10': [10000.0, 10500.0]
    })
    
    # Predict using the model
    predictions = model.predict(dummy_data)
    
    # Check that predictions are valid (0 or 1 for classification)
    assert predictions.shape[0] == 2
    assert all(pred in [0, 1] for pred in predictions)
    
    print("test_model_loading_and_prediction PASSED")

if __name__ == "__main__":
    test_model_loading_and_prediction()
