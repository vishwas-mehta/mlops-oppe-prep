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
        'timestamp': pd.to_datetime(['2023-01-01 09:00:00', '2023-01-01 09:01:00', 
                                     '2023-01-01 09:05:00', '2023-01-01 09:06:00']),
        'close': [100, 101, 102, 99],
        'volume': [1000]*4
    }
    df = pd.DataFrame(data)
    
    # Manually compute what the target should be
    # For timestamp 09:00:00, future close at 09:05:00 is 102. 102 > 100, so target=1
    # For timestamp 09:01:00, future close at 09:06:00 is 99. 99 < 101, so target=0
    # Add dummy rows to make shift(-5) work
    full_df = pd.concat([df, pd.DataFrame({'timestamp': pd.to_datetime(['2023-01-01 09:02:00', '2023-01-01 09:03:00', '2023-01-01 09:04:00']), 'close': [0,0,0], 'volume': [0,0,0]})])
    
    processed_df = create_features_and_target(full_df)
    
    assert processed_df[processed_df['close'] == 100]['target'].iloc[0] == 1
    assert processed_df[processed_df['close'] == 101]['target'].iloc[0] == 0
    print("test_target_variable PASSED")

if __name__ == "__main__":
    test_feature_creation_columns()
    test_target_variable()
