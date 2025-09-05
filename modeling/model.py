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
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor, BaggingRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns

# Import the time-series cross-validation utility
from cross_validation.train_test_split import prepare_tscv_splits


class EnsembleWrapper:
    """Wrapper class to handle ensemble model predictions"""
    def __init__(self, stack_model, vote_model, weights=(0.6, 0.4)):
        self.stack_model = stack_model
        self.vote_model = vote_model
        self.weights = weights
    
    def predict(self, X):
        """Make predictions using weighted average of stack and vote models"""
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
        stack_pred = self.stack_model.predict(X_array)
        vote_pred = self.vote_model.predict(X_array)
        return self.weights[0] * stack_pred + self.weights[1] * vote_pred


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
    print("\nrunning model comparison")

    # Define a dictionary of models to test with expanded options
    models_to_test = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
        'XGBoost': XGBRegressor(random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'Neural Network': MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
    }

    print("models:")
    print("random forest: tree based")
    print("lightgbm: gradient boosting")
    print("xgboost: gradient boosting") 
    print("svr: support vector regression")
    print("neural network: mlp")

    print("evaluating models...")
    all_results = {name: [] for name in models_to_test}

    # Run the time-series cross-validation loop
    for i, (X_train, X_test, y_train, y_test) in enumerate(prepare_tscv_splits(df, features, target_variable, n_splits=n_splits)):
        print(f"\n--- Running Fold {i+1}/{n_splits} ---")
        
        # Evaluate all models on the current fold
        fold_results = evaluate_multiple_models(X_train, y_train, X_test, y_test, models_to_test)
        
        # Append the results of this fold to our master results dictionary
        for name, mse in fold_results.items():
            all_results[name].append(mse)

    # Calculate and sort average performance for each model
    model_performances = {}
    print("\nmodel ranking by performance:")
    for name, results in all_results.items():
        avg_rmse = np.sqrt(np.mean(results))
        model_performances[name] = {
            'avg_rmse': avg_rmse,
            'mses': results,
            'std_rmse': np.std([np.sqrt(mse) for mse in results])
        }
    
    # Sort models by average RMSE
    sorted_models = sorted(model_performances.items(), key=lambda x: x[1]['avg_rmse'])
    
    # Print detailed performance comparison
    for rank, (name, perf) in enumerate(sorted_models, 1):
        print(f"\n{rank}. {name}")
        print(f"   average rmse: {perf['avg_rmse']:.4f}")
        print(f"   std rmse: {perf['std_rmse']:.4f}")
        print(f"   mse by fold: {[f'{mse:.4f}' for mse in perf['mses']]}")
    
    return sorted_models[0][0], model_performances  # Return best model name and all performances


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
    """Creates an enhanced stacking ensemble with optimized base models"""
    # Enhanced base models with better configurations
    estimators = [
        ('lgb', LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=8,
            num_leaves=100,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            verbose=-1,
            importance_type='gain'
        )),
        ('xgb', XGBRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            tree_method='hist',
            grow_policy='lossguide'
        )),
        ('cat', CatBoostRegressor(
            iterations=3000,
            learning_rate=0.01,
            depth=8,
            l2_leaf_reg=3,
            random_strength=0.1,
            verbose=False,
            random_state=random_state
        )),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(200, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            random_state=random_state
        )),
        ('svr', SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1,
            gamma='scale'
        )),
        ('kr', KernelRidge(
            alpha=0.1,
            kernel='rbf',
            gamma=0.1
        )),
        ('bag', BaggingRegressor(
            estimator=LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.01
            ),
            n_estimators=10,
            max_samples=0.85,
            max_features=0.85,
            random_state=random_state
        ))
    ]
    
    # Create enhanced stacking regressor
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=50,
            random_state=random_state
        ),
        cv=5,
        n_jobs=-1,
        passthrough=True  # Include original features
    )
    
    # Create weighted voting ensemble with optimized weights
    voter = VotingRegressor(
        estimators=estimators,
        weights=[0.25, 0.25, 0.2, 0.1, 0.1, 0.05, 0.05]  # Adjusted weights
    )
    
    return stack, voter


def tune_model_stack(X_train, y_train, X_test, y_test, random_state=42):
    print("\ntraining ensemble models")
    
    # Get both ensemble models
    stack_model, vote_model = create_advanced_model_stack(random_state)
    
    # Convert to numpy arrays
    X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    
    # Train both models
    stack_model.fit(X_train_array, y_train)
    vote_model.fit(X_train_array, y_train)
    
    # Get predictions from both models
    stack_pred = stack_model.predict(X_test_array)
    vote_pred = vote_model.predict(X_test_array)
    
    # Calculate MSE for both
    stack_mse = mean_squared_error(y_test, stack_pred)
    vote_mse = mean_squared_error(y_test, vote_pred)
    
    # Calculate optimal weights using validation performance
    weights = np.array([0.6, 0.4])  # Initial weights
    
    # Simple gradient descent to optimize weights
    learning_rate = 0.01
    n_iterations = 100
    
    for _ in range(n_iterations):
        # Calculate weighted prediction
        weighted_pred = (weights[0] * stack_pred + weights[1] * vote_pred)
        current_mse = mean_squared_error(y_test, weighted_pred)
        
        # Calculate gradients
        grad_stack = -2 * np.mean((y_test - weighted_pred) * stack_pred)
        grad_vote = -2 * np.mean((y_test - weighted_pred) * vote_pred)
        gradients = np.array([grad_stack, grad_vote])
        
        # Update weights
        weights -= learning_rate * gradients
        weights = np.clip(weights, 0, 1)  # Ensure weights are between 0 and 1
        weights /= np.sum(weights)  # Normalize weights to sum to 1
    
    # Use optimized weights for final prediction
    final_pred = weights[0] * stack_pred + weights[1] * vote_pred
    final_mse = mean_squared_error(y_test, final_pred)
    
    print(f"optimized weights: {weights}")
    print(f"stack mse: {stack_mse:.4f}")
    print(f"vote mse: {vote_mse:.4f}")
    print(f"combined mse: {final_mse:.4f}")
    
    if final_mse < min(stack_mse, vote_mse):
        print("selected: combined ensemble")
        ensemble = EnsembleWrapper(stack_model, vote_model)
        return ensemble, final_mse
    elif stack_mse < vote_mse:
        print("selected: stack only")
        return stack_model, stack_mse
    else:
        print("selected: vote only")
        return vote_model, vote_mse


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


def make_ensemble_prediction(models, X):
    """Helper function to make predictions using ensemble models"""
    if isinstance(models, tuple):
        # If we have both stack and vote models
        stack_model, vote_model = models
        stack_pred = stack_model.predict(X)
        vote_pred = vote_model.predict(X)
        # Return weighted average
        return 0.6 * stack_pred + 0.4 * vote_pred
    else:
        # Single model
        return models.predict(X)

def plot_holdout_predictions(holdout_results, model_name):
    """
    Creates a line plot comparing actual vs predicted values on holdout set.
    """
    plt.figure(figsize=(12, 6))
    
    # Get actual and predicted values
    y_true = holdout_results['actual']
    y_pred = holdout_results['predictions']
    
    # Create line plot
    plt.plot(range(len(y_true)), y_true, label='Actual', color='blue', linewidth=2)
    plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
    
    # Add error bands (standard deviation of residuals)
    std_residuals = np.std(holdout_results['residuals'])
    plt.fill_between(
        range(len(y_true)),
        y_pred - std_residuals,
        y_pred + std_residuals,
        color='red',
        alpha=0.1,
        label='±1 STD'
    )
    
    plt.title(f'Actual vs Predicted Values ({model_name})\nHoldout Set Performance')
    plt.xlabel('Sample Index')
    plt.ylabel('Energy Consumption')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add performance metrics as text
    metrics_text = (
        f"RMSE: {holdout_results['rmse']:.4f}\n"
        f"MAE: {holdout_results['mae']:.4f}\n"
        f"MAPE: {holdout_results['mape']:.2f}%"
    )
    plt.text(
        0.02, 0.98, metrics_text,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    return plt.gcf()

def make_holdout_predictions(model, X_holdout, y_holdout, model_name="Model"):
    """Make predictions on the completely unseen holdout dataset."""
    print(f"\ntesting {model_name} on holdout data")
    
    # Handle numpy conversion for ensemble models
    if hasattr(X_holdout, 'values'):
        X_holdout_array = X_holdout.values
    else:
        X_holdout_array = X_holdout
    
    # Make predictions using the helper function
    y_holdout_pred = make_ensemble_prediction(model, X_holdout_array)
    
    # Calculate metrics
    holdout_mse = mean_squared_error(y_holdout, y_holdout_pred)
    holdout_rmse = np.sqrt(holdout_mse)
    mae = np.mean(np.abs(y_holdout - y_holdout_pred))
    mape = np.mean(np.abs((y_holdout - y_holdout_pred) / y_holdout)) * 100
    residuals = y_holdout - y_holdout_pred  # Calculate residuals first
    
    # Create visualization with all metrics available
    fig = plot_holdout_predictions(
        {
            'predictions': y_holdout_pred,
            'actual': y_holdout,
            'rmse': holdout_rmse,
            'mae': mae,
            'mape': mape,
            'residuals': residuals
        },
        model_name
    )
    
    # Save the plot
    os.makedirs('visualization_results', exist_ok=True)
    fig.savefig(f'visualization_results/holdout_predictions_{model_name.lower().replace(" ", "_")}.png')
    plt.close(fig)
    
    # Print metrics
    print(f"holdout performance:")
    print(f"  rmse: {holdout_rmse:.4f}")
    print(f"  mse: {holdout_mse:.4f}")
    print(f"  mae: {mae:.4f}")
    print(f"  mape: {mape:.2f}%")
    
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
                                  holdout_results, feature_names, scaler):
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
        'scaler': scaler,
        'model_type': 'LightGBM',
        'timestamp': timestamp
    }, lgb_filename)
    
    # Save Stacking model with special handling for tuples
    stack_filename = f'saved_models/stacking_model_{timestamp}.joblib'
    stack_data = {
        'model': stack_model,
        'test_rmse': stack_rmse,
        'holdout_results': holdout_results.get('stack', {}),
        'feature_names': feature_names,
        'scaler': scaler,
        'model_type': 'Stacking_Ensemble',
        'timestamp': timestamp,
        'is_ensemble_wrapper': isinstance(stack_model, EnsembleWrapper)
    }
    joblib.dump(stack_data, stack_filename)
    
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


def make_prediction(model_data, X):
    """Helper function to make predictions using loaded models"""
    model = model_data['model']
    if model_data.get('is_ensemble_wrapper', False):
        # Handle ensemble wrapper models
        ensemble = model
        return ensemble.predict(X)
    else:
        # Single model
        return model.predict(X)

def load_model_for_prediction(model_path):
    """Load a saved model and its associated scaler for prediction"""
    model_data = joblib.load(model_path)
    print(f"Loaded {model_data['model_type']} model")
    print(f"  Test RMSE: {model_data['test_rmse']:.4f}")
    
    if 'holdout_results' in model_data and model_data['holdout_results']:
        holdout_rmse = model_data['holdout_results']['rmse']
        print(f"  Holdout RMSE: {holdout_rmse:.4f}")
    if 'scaler' in model_data:
        print("  Associated feature scaler also loaded.")
    
    return model_data


def run_final_model_training_with_holdout(X_train, y_train, X_test, y_test, X_holdout, y_holdout, feature_names, scaler):
    print("\nrunning final model training")
    
    # Initialize all models
    models_to_evaluate = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
        'XGBoost': XGBRegressor(random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'Neural Network': MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
    }
    
    # Train and evaluate each model
    model_results = {}
    for name, model in models_to_evaluate.items():
        print(f"\ntraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        model_results[name] = {'model': model, 'test_rmse': rmse}
    
    # Sort by performance
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['test_rmse'])
    
    print("\nmodel performance ranking:")
    for rank, (name, results) in enumerate(sorted_models, 1):
        print(f"{rank}. {name}: {results['test_rmse']:.4f} rmse")
    
    # Select best model
    best_model_name, best_results = sorted_models[0]
    best_model = best_results['model']
    
    print(f"\nbest model: {best_model_name}")
    
    # Also train stacking ensemble for comparison
    print("\ntraining stacking ensemble for comparison...")
    stack_model, stack_rmse = tune_model_stack(X_train, y_train, X_test, y_test)
    
    # Compare best individual model with stacking ensemble
    if stack_rmse < best_results['test_rmse']:
        print("stacking ensemble performs better than individual models")
        final_model = stack_model
        final_model_name = "Stacking Ensemble"
    else:
        print(f"{best_model_name} performs better than stacking ensemble")
        final_model = best_model
        final_model_name = best_model_name
    
    # Step 4: NOW use the holdout dataset for final validation
    print("\nHOLDOUT DATASET VALIDATION (COMPLETELY UNSEEN DATA)")
    print("  This is the FIRST TIME these models see the holdout data!")
    
    # Test models on holdout data
    holdout_results = {}
    
    print("\ntesting individual model on holdout data...")
    holdout_results['individual'] = make_holdout_predictions(
        final_model, X_holdout, y_holdout, final_model_name
    )
    
    print("\ntesting stacking ensemble on holdout data...")
    holdout_results['stack'] = make_holdout_predictions(
        stack_model, X_holdout, y_holdout, "Stacking Ensemble"
    )
    
    # Get holdout RMSEs
    individual_holdout_rmse = holdout_results['individual']['rmse']
    stack_holdout_rmse = holdout_results['stack']['rmse']
    
    # Final performance comparison
    print("\nfinal performance comparison")
    print(f"{'Model':<20} {'Test RMSE':<12} {'Holdout RMSE':<15} {'Difference':<12}")
    print(f"{'-'*65}")
    
    if final_model_name == "Stacking Ensemble":
        # Show best individual model first, then stacking ensemble
        print(f"{best_model_name:<20} {best_results['test_rmse']:<12.4f} {individual_holdout_rmse:<15.4f} {abs(best_results['test_rmse'] - individual_holdout_rmse):<12.4f}")
        print(f"{'Stacking Ensemble':<20} {stack_rmse:<12.4f} {stack_holdout_rmse:<15.4f} {abs(stack_rmse - stack_holdout_rmse):<12.4f}")
    else:
        # Show stacking ensemble first, then best individual model
        print(f"{'Stacking Ensemble':<20} {stack_rmse:<12.4f} {stack_holdout_rmse:<15.4f} {abs(stack_rmse - stack_holdout_rmse):<12.4f}")
        print(f"{best_model_name:<20} {best_results['test_rmse']:<12.4f} {individual_holdout_rmse:<15.4f} {abs(best_results['test_rmse'] - individual_holdout_rmse):<12.4f}")

    # Calculate generalization gaps
    individual_generalization_gap = abs(best_results['test_rmse'] - individual_holdout_rmse)
    stack_generalization_gap = abs(stack_rmse - stack_holdout_rmse)
    
    print("\ngeneralization analysis:")
    # Report for best individual model
    if individual_generalization_gap < 0.01:
        print(f"    {best_model_name}: excellent generalization (gap: {individual_generalization_gap:.4f})")
    elif individual_generalization_gap < 0.05:
        print(f"    {best_model_name}: good generalization (gap: {individual_generalization_gap:.4f})")
    else:
        print(f"    {best_model_name}: poor generalization (gap: {individual_generalization_gap:.4f})")
    
    # Report for stacking ensemble
    if stack_generalization_gap < 0.01:
        print(f"    Stacking Ensemble: excellent generalization (gap: {stack_generalization_gap:.4f})")
    elif stack_generalization_gap < 0.05:
        print(f"    Stacking Ensemble: good generalization (gap: {stack_generalization_gap:.4f})")
    else:
        print(f"    Stacking Ensemble: poor generalization (gap: {stack_generalization_gap:.4f})")
    
    # Save results
    print("\nsaving models and results...")
    results_file = save_model_results(
        final_model, stack_model, 
        best_results['test_rmse'], stack_rmse,
        holdout_results, feature_names, scaler
    )
    
    return final_model, final_model_name, holdout_results, {
        'best_model': best_model,
        'best_test_rmse': best_results['test_rmse'],
        'best_holdout_rmse': individual_holdout_rmse,
        'stack_model': stack_model,
        'stack_test_rmse': stack_rmse,
        'stack_holdout_rmse': stack_holdout_rmse,
        'model_files': {'results': results_file}
    }

def save_model_results(final_model, stack_model, final_rmse, stack_rmse, holdout_results, feature_names, scaler):
    """Helper function to save model results"""
    os.makedirs('saved_models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_filename = f'saved_models/model_comparison_and_holdout_{timestamp}.txt'
    with open(results_filename, 'w') as f:
        f.write(f"Model Comparison and Holdout Results - {timestamp}\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL SELECTION RESULTS:\n")
        f.write(f"Final Model Test RMSE: {final_rmse:.4f}\n")
        f.write(f"Stacking Test RMSE: {stack_rmse:.4f}\n\n")
        
        f.write("HOLDOUT VALIDATION RESULTS:\n")
        for model_key, results in holdout_results.items():
            f.write(f"{results['model_name']}:\n")
            f.write(f"  RMSE: {results['rmse']:.4f}\n")
            f.write(f"  MAE: {results['mae']:.4f}\n")
            f.write(f"  MAPE: {results['mape']:.2f}%\n\n")
        
        f.write(f"Feature Names: {feature_names}\n")
    
    return results_filename