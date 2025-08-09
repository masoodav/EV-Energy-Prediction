##############################################################
# Import main libraries 
##############################################################
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer

def find_best_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Refines the search for the best RandomForestRegressor model using a focused
    GridSearchCV around a known set of good hyperparameters.

    This function is designed to be used after an initial broad search
    (e.g., with RandomizedSearchCV) to fine-tune the model.

    Args:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.Series): The training target variable.
        X_test (pd.DataFrame): The testing feature set.
        y_test (pd.Series): The testing target variable.

    Returns:
        tuple: A tuple containing the best model found (sklearn estimator), its best
               parameters (dict), and its performance (float) on the test set.
    """
    print("\n--- Running a Focused GridSearchCV to Refine the Model ---")

    # Initialize the model and scoring metric
    rf = RandomForestRegressor(random_state=42)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Define a more focused parameter grid around the best parameters
    # you have already identified. We'll search a small range of values
    # for each parameter to find the optimal combination.
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [8, 10, 12, None],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'], # Keep this fixed as it's the best value
        'min_impurity_decrease': [0.0, 0.01, 0.02]
    }
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring=mse_scorer,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)

    # Get the best model and its results from the grid search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate the best model on the test set
    test_mse = mean_squared_error(y_test, best_model.predict(X_test))

    print(f"\n  ✅ Refined GridSearchCV Best Parameters: {best_params}")
    print(f"  Refined Model Test MSE: {test_mse:.4f}")

    return best_model, best_params, test_mse

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

