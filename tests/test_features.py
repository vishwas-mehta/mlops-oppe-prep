import pandas as pd
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from features import create_features_and_target

def test_feature_creation_columns():
    """Test if create_features_and_target produces the expected columns."""
    data = {
        'timestamp': pd.to_datetime([
            '2023-01-01 09:00:00', '2023-01-01 09:01:00', '2023-01-01 09:02:00',
            '2023-01-01 09:03:00', '2023-01-01 09:04:00', '2023-01-01 09:05:00',
            '2023-01-01 09:06:00', '2023-01-01 09:07:00', '2023-01-01 09:08:00',
            '2023-01-01 09:09:00', '2023-01-01 09:10:00', '2023-01-01 09:11:00',
            '2023-01-01 09:12:00', '2023-01-01 09:13:00', '2023-01-01 09:14:00',
            '2023-01-01 09:15:00'
        ]),
        'open': [100]*16, 'high': [102]*16, 'low': [99]*16,
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 100, 100, 100, 100, 100],
        'volume': [1000]*16
    }
    df = pd.DataFrame(data)
    
    processed_df = create_features_and_target(df)
    
    expected_columns = ['rolling_avg_10', 'volume_sum_10', 'target']
    
    assert all(col in processed_df.columns for col in expected_columns)
    print("test_feature_creation_columns PASSED")

def test_target_variable():
    """Test the logic of the target variable creation."""
    data = {
        'timestamp': pd.to_datetime([
            '2023-01-01 09:00:00', '2023-01-01 09:01:00', '2023-01-01 09:02:00',
            '2023-01-01 09:03:00', '2023-01-01 09:04:00', '2023-01-01 09:05:00',
            '2023-01-01 09:06:00', '2023-01-01 09:07:00', '2023-01-01 09:08:00',
            '2023-01-01 09:09:00', '2023-01-01 09:10:00', '2023-01-01 09:11:00',
            '2023-01-01 09:12:00', '2023-01-01 09:13:00', '2023-01-01 09:14:00',
            '2023-01-01 09:15:00'
        ]),
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 109, 108, 107, 106, 105],
        'volume': [1000]*16
    }
    df = pd.DataFrame(data)
    
    processed_df = create_features_and_target(df)
    
    assert 'target' in processed_df.columns
    assert processed_df['target'].isin([0, 1]).all()
    assert len(processed_df) > 0
    
    print("test_target_variable PASSED")

if __name__ == "__main__":
    test_feature_creation_columns()
    test_target_variable()
