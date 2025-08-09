##############################################################
# Import main libraries 
##############################################################
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def prepare_tscv_splits(df: pd.DataFrame, features: list, target: str, n_splits: int = 5):
    """
    Prepares a time-series cross-validation split of the data.

    This function generates train/test indices for each fold without
    training a model. This is the correct approach for time-series data
    to prevent data leakage.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        features (list): The list of feature columns.
        target (str): The name of the target variable.
        n_splits (int): The number of splits for cross-validation.

    Yields:
        tuple: A tuple containing the following for each fold:
               X_train (pd.DataFrame), X_test (pd.DataFrame),
               y_train (pd.DataFrame), y_test (pd.DataFrame)
    """
    print(f"\n--- Preparing Time-Series Cross-Validation Splits with {n_splits} folds ---")
    
    # Define features (X) and target (y)
    X = df[features]
    y = df[target]

    # Initialize the time-series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Iterate through each split and yield the dataframes
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        # Split the data into training and testing sets for this fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        print(f"  Fold {i+1} prepared:")
        print(f"    Train data size: {len(X_train)} samples")
        print(f"    Test data size: {len(X_test)} samples")
        
        yield X_train, X_test, y_train, y_test

