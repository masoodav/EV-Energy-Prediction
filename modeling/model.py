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


def prepare_train_test_holdout_splits(df: pd.DataFrame, features: list, target_variable: str, 
                                    train_ratio: float = 0.7, test_ratio: float = 0.2):
    """
    Prepares train/test/holdout splits for time series data maintaining chronological order.
    The holdout set is completely reserved for final model validation after model selection.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame containing features and target.
        features (list): A list of feature column names.
        target_variable (str): The name of the target column.
        train_ratio (float): Proportion of data for training (default 0.7).
        test_ratio (float): Proportion of data for testing/model selection (default 0.2).
                          Holdout ratio will be 1 - train_ratio - test_ratio.
    
    Returns:
        tuple: (X_train, X_test, X_holdout, y_train, y_test, y_holdout)
    """
    n_samples = len(df)
    train_end = int(n_samples * train_ratio)
    test_end = int(n_samples * (train_ratio + test_ratio))
    
    # Split data chronologically
    train_data = df.iloc[:train_end]
    test_data = df.iloc[train_end:test_end]
    holdout_data = df.iloc[test_end:]
    
    # Prepare feature and target sets
    X_train = train_data[features]
    X_test = test_data[features]
    X_holdout = holdout_data[features]
    
    y_train = train_data[target_variable]
    y_test = test_data[target_variable]
    y_holdout = holdout_data[target_variable]
    
    holdout_ratio = 1 - train_ratio - test_ratio
    
    print(f"Data split sizes:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/n_samples:.1%}) - for model training")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/n_samples:.1%}) - for model selection & tuning")
    print(f"  Holdout: {len(X_holdout)} samples ({len(X_holdout)/n_samples:.1%}) - RESERVED for final validation")
    
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout


def run_model_comparison_pipeline(df: pd.DataFrame, features: list, target_variable: str, n_splits: int = 5):
    """
    Orchestrates the entire model comparison pipeline using time-series cross-validation.
    This uses only the train+test portion of the data (holdout is completely reserved).

    Args:
        df (pd.DataFrame): The preprocessed DataFrame containing features and target (WITHOUT holdout data).
        features (list): A list of feature column names.
        target_variable (str): The name of the target column.
        n_splits (int): The number of time-series cross-validation splits to use.
    """
    print("\n--- Time-Series Cross-Validation and Model Training (Holdout Data Excluded) ---")

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
        print(f"  Average Test RMSE: {np.sqrt(np.mean(results)):.4f}")


def test_different_random_states(X_train, y_train, X_test, y_test, random_states=[42, 123, 555, 777, 999]):
    """
    Tests model performance with different random states to ensure robustness.
    Uses only train/test data - holdout is completely reserved.
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
            learning_rate=0.05,
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
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state
        )),
        ('cat', CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
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
    Uses only train/test data - holdout is reserved.
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
    Enhanced LightGBM tuning with improved parameters and learning rate ranges.
    Uses only train/test data - holdout is reserved.
    """
    print(f"\n--- Quick Tuning LightGBM (random_state={random_state}) ---")
    
    # Initialize base model
    lgbm = LGBMRegressor(
        random_state=random_state,
        verbose=-1,
        objective='regression',
    )
    
    # Enhanced parameter space with higher learning rates
    param_distributions = {
        'n_estimators': [1000, 1500, 2000],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
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
        n_iter=25,
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
    
    # Fit with early stopping using test set for evaluation
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # Evaluate final model
    y_pred = final_model.predict(X_test)
    final_rmse = rmse_scorer(y_test, y_pred)
    
    print(f"\n Best Parameters: {best_params}")
    print(f"Final RMSE: {final_rmse:.4f}")
    
    return final_model, best_params, final_rmse


def make_holdout_predictions(model, X_holdout, y_holdout, model_name="Model"):
    """
    Make predictions on the completely unseen holdout dataset.
    
    Args:
        model: Trained model
        X_holdout: Holdout features
        y_holdout: Holdout target values
        model_name: Name of the model for reporting
    
    Returns:
        dict: Dictionary containing predictions and metrics
    """
    print(f"\n--- Making Predictions on Holdout Dataset with {model_name} ---")
    
    # Handle numpy conversion for ensemble models
    if hasattr(X_holdout, 'values'):
        X_holdout_array = X_holdout.values
    else:
        X_holdout_array = X_holdout
    
    # Make predictions
    y_holdout_pred = model.predict(X_holdout_array)
    
    # Calculate metrics
    holdout_mse = mean_squared_error(y_holdout, y_holdout_pred)
    holdout_rmse = np.sqrt(holdout_mse)
    
    # Calculate additional metrics
    mae = np.mean(np.abs(y_holdout - y_holdout_pred))
    mape = np.mean(np.abs((y_holdout - y_holdout_pred) / y_holdout)) * 100
    
    print(f" HOLDOUT DATASET PERFORMANCE ({model_name}):")
    print(f"   RMSE: {holdout_rmse:.4f}")
    print(f"   MSE: {holdout_mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    # Residual analysis
    residuals = y_holdout - y_holdout_pred
    print(f"   Mean Residual: {np.mean(residuals):.4f}")
    print(f"   Std Residual: {np.std(residuals):.4f}")
    print(f"   Min Residual: {np.min(residuals):.4f}")
    print(f"   Max Residual: {np.max(residuals):.4f}")
    
    return {
        'predictions': y_holdout_pred,
        'actual': y_holdout,
        'rmse': holdout_rmse,
        'mse': holdout_mse,
        'mae': mae,
        'mape': mape,
        'residuals': residuals,
        'model_name': model_name
    }


def save_models_and_holdout_results(lgb_model, stack_model, lgb_params, lgb_rmse, stack_rmse, 
                                  holdout_results, feature_names):
    """
    Save both models and holdout validation results for production deployment.
    """
    # Create models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save LightGBM model
    lgb_filename = f'saved_models/lightgbm_model_{timestamp}.joblib'
    joblib.dump({
        'model': lgb_model,
        'params': lgb_params,
        'test_rmse': lgb_rmse,
        'holdout_results': holdout_results.get('lgb', {}),
        'feature_names': feature_names,
        'model_type': 'LightGBM',
        'timestamp': timestamp
    }, lgb_filename)
    
    # Save Stacking model
    stack_filename = f'saved_models/stacking_model_{timestamp}.joblib'
    joblib.dump({
        'model': stack_model,
        'test_rmse': stack_rmse,
        'holdout_results': holdout_results.get('stack', {}),
        'feature_names': feature_names,
        'model_type': 'Stacking_Ensemble',
        'timestamp': timestamp
    }, stack_filename)
    
    # Save comprehensive comparison and holdout results
    results_filename = f'saved_models/model_comparison_and_holdout_{timestamp}.txt'
    with open(results_filename, 'w') as f:
        f.write(f"Model Comparison and Holdout Validation Results - {timestamp}\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL SELECTION RESULTS (Test Set):\n")
        f.write("-"*40 + "\n")
        f.write(f"LightGBM Test RMSE: {lgb_rmse:.4f}\n")
        f.write(f"Stacking Ensemble Test RMSE: {stack_rmse:.4f}\n\n")
        
        # Determine best model
        best_model_name = 'Stacking Ensemble' if stack_rmse < lgb_rmse else 'LightGBM'
        performance_diff = abs(lgb_rmse - stack_rmse)
        f.write(f"Selected Model: {best_model_name}\n")
        f.write(f"Performance Difference: {performance_diff:.4f} RMSE\n\n")
        
        f.write("HOLDOUT VALIDATION RESULTS (Completely Unseen Data):\n")
        f.write("-"*50 + "\n")
        
        for model_key, results in holdout_results.items():
            model_name = results['model_name']
            f.write(f"{model_name}:\n")
            f.write(f"  RMSE: {results['rmse']:.4f}\n")
            f.write(f"  MAE: {results['mae']:.4f}\n")
            f.write(f"  MAPE: {results['mape']:.2f}%\n")
            f.write(f"  Mean Residual: {np.mean(results['residuals']):.4f}\n")
            f.write(f"  Std Residual: {np.std(results['residuals']):.4f}\n\n")
        
        f.write(f"Feature Names: {feature_names}\n\n")
        f.write(f"LightGBM Parameters: {lgb_params}\n")
    
    print(f"\nModels and holdout results saved:")
    print(f"   LightGBM: {lgb_filename}")
    print(f"   Stacking: {stack_filename}")
    print(f"   Results: {results_filename}")
    
    return lgb_filename, stack_filename, results_filename


def load_model_for_prediction(model_path):
    """
    Load a saved model for prediction
    """
    model_data = joblib.load(model_path)
    print(f"Loaded {model_data['model_type']} model")
    print(f"  Test RMSE: {model_data['test_rmse']:.4f}")
    if 'holdout_results' in model_data and model_data['holdout_results']:
        holdout_rmse = model_data['holdout_results']['rmse']
        print(f"  Holdout RMSE: {holdout_rmse:.4f}")
    return model_data


def run_final_model_training_with_holdout(X_train, y_train, X_test, y_test, X_holdout, y_holdout, feature_names):
    """
    Run the complete model training pipeline with holdout validation.
    
    The holdout dataset is ONLY used at the very end for final validation predictions.
    """
    print("\n" + "="*80)
    print("FINAL MODEL TRAINING PIPELINE WITH HOLDOUT VALIDATION")
    print("="*80)
    
    # Step 1: Find best random state using only train/test data
    print("\nStep 1: Finding optimal random state (using train/test data only)...")
    best_random_state = test_different_random_states(X_train, y_train, X_test, y_test)
    
    # Step 2: Train individual LightGBM using only train/test data
    print(f"\nStep 2: Training LightGBM with optimal random state ({best_random_state})...")
    lgb_model, lgb_params, lgb_rmse = tune_lightgbm_model(
        X_train, y_train, X_test, y_test, 
        random_state=best_random_state
    )
    
    # Step 3: Train stacking ensemble using only train/test data
    print(f"\nStep 3: Training Stacking Ensemble...")
    stack_model, stack_rmse = tune_model_stack(
        X_train, y_train, X_test, y_test,
        random_state=best_random_state
    )
    
    # Step 4: Select best model based on test performance
    print("\n" + "="*80)
    print("MODEL SELECTION RESULTS (Test Set)")
    print("="*80)
    print(f"LightGBM Test RMSE:      {lgb_rmse:.4f}")
    print(f"Stacking Test RMSE:      {stack_rmse:.4f}")
    
    improvement = abs(lgb_rmse - stack_rmse)
    improvement_pct = (improvement / max(lgb_rmse, stack_rmse)) * 100
    
    if stack_rmse < lgb_rmse:
        print(f"🏆 Stacking Ensemble selected (improvement: {improvement:.4f} RMSE, {improvement_pct:.1f}%)")
        best_model = stack_model
        best_model_name = "Stacking Ensemble"
    else:
        print(f"🏆 LightGBM selected (improvement: {improvement:.4f} RMSE, {improvement_pct:.1f}%)")
        best_model = lgb_model
        best_model_name = "LightGBM"
    
    # Step 5: NOW use the holdout dataset for final validation
    print("\n" + "="*80)
    print("HOLDOUT DATASET VALIDATION (COMPLETELY UNSEEN DATA)")
    print("="*80)
    print("  This is the FIRST TIME these models see the holdout data!")
    
    # Test both models on holdout data
    holdout_results = {}
    
    print(f"\nTesting LightGBM on holdout data...")
    holdout_results['lgb'] = make_holdout_predictions(
        lgb_model, X_holdout, y_holdout, "LightGBM"
    )
    
    print(f"\nTesting Stacking Ensemble on holdout data...")
    holdout_results['stack'] = make_holdout_predictions(
        stack_model, X_holdout, y_holdout, "Stacking Ensemble"
    )
    
    # Step 6: Final performance comparison
    print("\n" + "="*80)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*80)
    
    lgb_holdout_rmse = holdout_results['lgb']['rmse']
    stack_holdout_rmse = holdout_results['stack']['rmse']
    
    print(f"{'Model':<20} {'Test RMSE':<12} {'Holdout RMSE':<15} {'Difference':<12}")
    print(f"{'-'*65}")
    print(f"{'LightGBM':<20} {lgb_rmse:<12.4f} {lgb_holdout_rmse:<15.4f} {abs(lgb_rmse - lgb_holdout_rmse):<12.4f}")
    print(f"{'Stacking Ensemble':<20} {stack_rmse:<12.4f} {stack_holdout_rmse:<15.4f} {abs(stack_rmse - stack_holdout_rmse):<12.4f}")
    
    # Generalization analysis
    lgb_generalization_gap = abs(lgb_rmse - lgb_holdout_rmse)
    stack_generalization_gap = abs(stack_rmse - stack_holdout_rmse)
    
    print(f"\n Generalization Analysis:")
    if lgb_generalization_gap < 0.01:
        print(f"    LightGBM: Excellent generalization (gap: {lgb_generalization_gap:.4f})")
    elif lgb_generalization_gap < 0.05:
        print(f"    LightGBM: Good generalization (gap: {lgb_generalization_gap:.4f})")
    else:
        print(f"     LightGBM: Poor generalization (gap: {lgb_generalization_gap:.4f})")
    
    if stack_generalization_gap < 0.01:
        print(f"    Stacking: Excellent generalization (gap: {stack_generalization_gap:.4f})")
    elif stack_generalization_gap < 0.05:
        print(f"    Stacking: Good generalization (gap: {stack_generalization_gap:.4f})")
    else:
        print(f"     Stacking: Poor generalization (gap: {stack_generalization_gap:.4f})")
    
    # Step 7: Save everything
    print(f"\nStep 7: Saving models and holdout results...")
    lgb_file, stack_file, results_file = save_models_and_holdout_results(
        lgb_model, stack_model, lgb_params, lgb_rmse, stack_rmse,
        holdout_results, feature_names
    )
    
    return best_model, best_model_name, holdout_results, {
        'lgb_model': lgb_model,
        'lgb_test_rmse': lgb_rmse,
        'lgb_holdout_rmse': lgb_holdout_rmse,
        'stack_model': stack_model,
        'stack_test_rmse': stack_rmse,
        'stack_holdout_rmse': stack_holdout_rmse,
        'model_files': {
            'lgb_file': lgb_file,
            'stack_file': stack_file,
            'results_file': results_file
        }
    }