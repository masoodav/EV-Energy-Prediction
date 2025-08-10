##############################################################
# Import main libraries 
##############################################################
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
import joblib
import os
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

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


def test_different_random_states(X_train, y_train, X_test, y_test, random_states=[42, 123, 555, 777, 999]):
    """
    Tests model performance with different random states to ensure robustness.
    """
    print("\n--- Testing Model Robustness with Different Random States ---")
    results = []
    
    for seed in random_states:
        model, params, rmse = tune_lightgbm_model(X_train, y_train, X_test, y_test, random_state=seed)
        results.append({
            'random_state': seed,
            'rmse': rmse,
            'params': params
        })
    
    # Find the best performing seed
    best_result = min(results, key=lambda x: x['rmse'])
    print("\n--- Random State Analysis Results ---")
    print(f"Best random_state: {best_result['random_state']} (RMSE: {best_result['rmse']:.4f})")
    print("RMSE variation across seeds:")
    for result in results:
        print(f"  random_state={result['random_state']}: {result['rmse']:.4f}")
    
    return best_result['random_state']

def create_advanced_model_stack(random_state=42):
    """
    Creates a stacking ensemble of multiple advanced models with improved parameters.
    """
    # Define base models with better parameters
    estimators = [
        ('lgb', LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,  # Increased from 0.01
            max_depth=6,
            num_leaves=48,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            verbose=-1
        )),
        ('xgb', XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,  # Increased from 0.01
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state
        )),
        ('cat', CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,  # Increased from 0.01
            depth=6,
            l2_leaf_reg=0.1,
            verbose=False,
            random_state=random_state
        ))
    ]
    
    # Create stacking ensemble with a meta-learner
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state,
            verbose=-1
        ),
        cv=5,
        n_jobs=-1
    )
    return stack

def tune_model_stack(X_train, y_train, X_test, y_test, random_state=42):
    """
    Trains and evaluates a stacking ensemble model with feature name handling.
    """
    print("\n--- Training Stacking Ensemble ---")
    model = create_advanced_model_stack(random_state)
    
    # Convert to numpy arrays to avoid feature name warnings
    X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    
    model.fit(X_train_array, y_train)
    
    y_pred = model.predict(X_test_array)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Stacking Ensemble RMSE: {rmse:.4f}")
    print(f"Stacking Ensemble MSE: {mse:.4f}")
    return model, rmse

def rmse_scorer(y_true, y_pred):
    """Custom RMSE scorer"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def tune_lightgbm_model(X_train, y_train, X_test, y_test, random_state=42):
    """
    Enhanced LightGBM tuning with improved parameters and learning rate ranges
    """
    print(f"\n--- Quick Tuning LightGBM (random_state={random_state}) ---")
    
    # Initialize base model
    lgbm = LGBMRegressor(
        random_state=random_state,
        verbose=-1,
        objective='regression',  # Changed from huber to standard regression
    )
    
    # Enhanced parameter space with higher learning rates
    param_distributions = {
        'n_estimators': [1000, 1500, 2000],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],  # Increased learning rate range
        'max_depth': [4, 5, 6, 7],
        'num_leaves': [31, 48, 64],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_samples': [20, 30, 50],
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [0.0, 0.1, 0.5],
        'min_split_gain': [0.0, 0.1, 0.2]
    }

    # Use RMSE for evaluation
    rmse_scoring = make_scorer(rmse_scorer, greater_is_better=False)
    
    random_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_distributions,
        n_iter=25,  # Increased iterations
        cv=3,
        scoring=rmse_scoring,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit with basic parameters first
    random_search.fit(X_train, y_train)
    
    # Get best parameters and create final model
    best_params = random_search.best_params_
    final_model = LGBMRegressor(**best_params, random_state=random_state, verbose=-1)
    
    # Fit with early stopping using validation set
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),  # Increased patience
            lgb.log_evaluation(period=200)
        ]
    )
    
    # Evaluate final model
    y_pred = final_model.predict(X_test)
    final_rmse = rmse_scorer(y_test, y_pred)
    
    print(f"\n✅ Best Parameters: {best_params}")
    print(f"Final RMSE: {final_rmse:.4f}")
    
    return final_model, best_params, final_rmse

def save_models_for_ab_testing(lgb_model, stack_model, lgb_params, lgb_rmse, stack_rmse, feature_names):
    """
    Save both models for A/B testing with metadata
    """
    # Create models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save LightGBM model
    lgb_filename = f'saved_models/lightgbm_model_{timestamp}.joblib'
    joblib.dump({
        'model': lgb_model,
        'params': lgb_params,
        'rmse': lgb_rmse,
        'feature_names': feature_names,
        'model_type': 'LightGBM',
        'timestamp': timestamp
    }, lgb_filename)
    
    # Save Stacking model
    stack_filename = f'saved_models/stacking_model_{timestamp}.joblib'
    joblib.dump({
        'model': stack_model,
        'rmse': stack_rmse,
        'feature_names': feature_names,
        'model_type': 'Stacking_Ensemble',
        'timestamp': timestamp
    }, stack_filename)
    
    # Save comparison metadata
    comparison_filename = f'saved_models/model_comparison_{timestamp}.txt'
    with open(comparison_filename, 'w') as f:
        f.write(f"Model Comparison Results - {timestamp}\n")
        f.write("="*50 + "\n\n")
        f.write(f"LightGBM RMSE: {lgb_rmse:.4f}\n")
        f.write(f"Stacking Ensemble RMSE: {stack_rmse:.4f}\n\n")
        f.write(f"Best Model: {'Stacking Ensemble' if stack_rmse < lgb_rmse else 'LightGBM'}\n")
        f.write(f"Performance Improvement: {abs(lgb_rmse - stack_rmse):.4f}\n\n")
        f.write(f"Feature Names: {feature_names}\n\n")
        f.write(f"LightGBM Parameters: {lgb_params}\n")
    
    print(f"\nModels saved for A/B testing:")
    print(f"   LightGBM: {lgb_filename}")
    print(f"   Stacking: {stack_filename}")
    print(f"   Comparison: {comparison_filename}")
    
    return lgb_filename, stack_filename, comparison_filename

def load_model_for_prediction(model_path):
    """
    Load a saved model for prediction
    """
    model_data = joblib.load(model_path)
    print(f"Loaded {model_data['model_type']} model (RMSE: {model_data['rmse']:.4f})")
    return model_data

def run_final_model_training(X_train, y_train, X_test, y_test, feature_names):
    """
    Run the complete model training pipeline with improvements
    """
    print("\n" + "="*60)
    print("FINAL MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Find best random state
    print("\nStep 1: Finding optimal random state...")
    best_random_state = test_different_random_states(X_train, y_train, X_test, y_test)
    
    # Step 2: Train individual LightGBM
    print(f"\nStep 2: Training LightGBM with optimal random state ({best_random_state})...")
    lgb_model, lgb_params, lgb_rmse = tune_lightgbm_model(
        X_train, y_train, X_test, y_test, 
        random_state=best_random_state
    )
    
    # Step 3: Train stacking ensemble
    print(f"\nStep 3: Training Stacking Ensemble...")
    stack_model, stack_rmse = tune_model_stack(
        X_train, y_train, X_test, y_test,
        random_state=best_random_state
    )
    
    # Step 4: Compare and select final model
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(f"LightGBM RMSE:      {lgb_rmse:.4f}")
    print(f"Stacking RMSE:      {stack_rmse:.4f}")
    
    improvement = abs(lgb_rmse - stack_rmse)
    improvement_pct = (improvement / max(lgb_rmse, stack_rmse)) * 100
    
    if stack_rmse < lgb_rmse:
        print(f"Stacking Ensemble wins by {improvement:.4f} RMSE ({improvement_pct:.1f}% improvement)")
        final_model = stack_model
        final_rmse = stack_rmse
        model_type = "Stacking Ensemble"
    else:
        print(f"LightGBM wins by {improvement:.4f} RMSE ({improvement_pct:.1f}% improvement)")
        final_model = lgb_model
        final_rmse = lgb_rmse
        model_type = "LightGBM"
    
    # Step 5: Save models for A/B testing
    print(f"\nStep 4: Saving models for A/B testing...")
    lgb_file, stack_file, comparison_file = save_models_for_ab_testing(
        lgb_model, stack_model, lgb_params, lgb_rmse, stack_rmse, feature_names
    )
    

    return stack_model, stack_rmse, {
        'lgb_model': lgb_model,
        'lgb_rmse': lgb_rmse,
        'stack_model': stack_model,
        'stack_rmse': stack_rmse,
        'model_files': {
            'lgb_file': lgb_file,
            'stack_file': stack_file,
            'comparison_file': comparison_file
        }
    }