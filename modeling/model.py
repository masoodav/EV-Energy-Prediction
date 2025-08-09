##############################################################
# Import main libraries 
##############################################################
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

def train_and_tune_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Trains a Random Forest Regressor and tunes its hyperparameters using RandomizedSearchCV.

    Args:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.Series): The training target variable.
        X_test (pd.DataFrame): The testing feature set.
        y_test (pd.Series): The testing target variable.

    Returns:
        tuple: A tuple containing the best model and the MSE on the test set.
    """
    print("\n--- Training and Tuning RandomForestRegressor ---")
    
    # Initialize the model
    rf = RandomForestRegressor(random_state=42)

    # Define the parameter distribution for RandomizedSearchCV
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=10,  # Number of parameter settings that are sampled
        cv=3,       # Number of cross-validation folds
        scoring='neg_mean_squared_error', # Use a regression scoring metric
        random_state=42,
        n_jobs=-1   # Use all available CPU cores
    )
    
    # Fit the random search to the training data
    random_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    print(f"  Best parameters found: {random_search.best_params_}")
    
    # Evaluate the best model on the test set
    y_preds = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_preds)
    print(f"  Test set MSE with best model: {mse:.4f}")
    
    return best_model, mse

