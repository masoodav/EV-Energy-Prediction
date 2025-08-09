##############################################################
# Import main libraries 
##############################################################
import pandas as pd
import numpy as np

# Import the models we will be comparing
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

# Import the time-series cross-validation utility
from cross_validation.train_test_split import prepare_tscv_splits


def evaluate_multiple_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, models_to_test: dict):
    """
    Trains and evaluates a dictionary of models on a single train/test split.

    Args:
        X_train (pd.DataFrame): Training feature set for the current fold.
        y_train (pd.Series): Training target variable for the current fold.
        X_test (pd.DataFrame): Testing feature set for the current fold.
        y_test (pd.Series): Testing target variable for the current fold.
        models_to_test (dict): A dictionary where keys are model names (str) and values
                               are initialized model estimators (e.g., RandomForestRegressor()).
    
    Returns:
        dict: A dictionary of mean squared errors (MSE) for each model on the test set.
    """
    mse_results = {}
    for name, model in models_to_test.items():
        # Train the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Calculate the Mean Squared Error and store it
        mse = mean_squared_error(y_test, y_pred)
        mse_results[name] = mse
        print(f"  {name} Test MSE: {mse:.4f}")
        
    return mse_results


def run_model_comparison_pipeline(df: pd.DataFrame, features: list, target_variable: str, n_splits: int = 5):
    """
    Orchestrates the entire model comparison pipeline using time-series cross-validation.

    This function sets up the models to be tested, runs the TSCV loop, and
    aggregates the results to provide a final performance comparison.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame containing features and target.
        features (list): A list of feature column names.
        target_variable (str): The name of the target column.
        n_splits (int): The number of time-series cross-validation splits to use.
    """
    print("\n--- Time-Series Cross-Validation and Model Training ---")

    # Define a dictionary of models to test
    models_to_test = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
        'XGBoost': XGBRegressor(random_state=42)
    }

    # Create a dictionary to store the results for each model
    all_results = {name: [] for name in models_to_test}

    # Run the time-series cross-validation loop
    for i, (X_train, X_test, y_train, y_test) in enumerate(prepare_tscv_splits(df, features, target_variable, n_splits=n_splits)):
        print(f"\n--- Running Fold {i+1}/{n_splits} ---")
        
        # Evaluate all models on the current fold
        fold_results = evaluate_multiple_models(X_train, y_train, X_test, y_test, models_to_test)
        
        # Append the results of this fold to our master results dictionary
        for name, mse in fold_results.items():
            all_results[name].append(mse)

    # Print combined results for comparison
    print("\n--- Combined Cross-Validation Results ---")
    for name, results in all_results.items():
        print(f"\n{name} Results:")
        print(f"  Test MSEs for each fold: {results}")
        print(f"  Average Test MSE: {np.mean(results):.4f}")


def find_best_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Refines the search for the best RandomForestRegressor model using a focused
    GridSearchCV around a known set of good hyperparameters.
    """
    print("\n--- Running a Focused GridSearchCV to Refine the Model ---")
    rf = RandomForestRegressor(random_state=42)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [8, 10, 12, None],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
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
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    test_mse = mean_squared_error(y_test, best_model.predict(X_test))

    print(f"\n  ✅ Refined GridSearchCV Best Parameters: {best_params}")
    print(f"  Refined Model Test MSE: {test_mse:.4f}")
    return best_model, best_params, test_mse

def train_and_tune_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Trains a Random Forest Regressor and tunes its hyperparameters using RandomizedSearchCV.
    """
    print("\n--- Training and Tuning RandomForestRegressor ---")
    rf = RandomForestRegressor(random_state=42)
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    print(f"  Best parameters found: {random_search.best_params_}")
    y_preds = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_preds)
    print(f"  Test set MSE with best model: {mse:.4f}")
    return best_model, mse
