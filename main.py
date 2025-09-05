##############################################################
# Import main libraries 
##############################################################
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import argparse

from data_collection.data_loader import load_kaggle_dataset
from data_preprocessing.data_preprocessor import (
    find_missing_data,
    suggest_imputation_strategy,
    auto_impute_missing_data,
    handle_outliers_iqr_method,
    scale_features_and_transform_target
)
from feature_engineering.feature_creator import (
    create_combined_ev_features,
    encode_categorical_features
)
from cross_validation.train_test_split import prepare_tscv_splits

# Updated imports from the improved model.py file
from modeling.model import (
    run_model_comparison_pipeline, 
    test_different_random_states,
    tune_model_stack,
    run_final_model_training_with_holdout,  # Updated function name
    save_models_and_holdout_results,
    load_model_for_prediction,
    prepare_train_test_holdout_splits,  # New function for 3-way split
    make_holdout_predictions
)

# Import all data analysis and visualization functions
from exploratory_data_analysis.data_analysis import (
    perform_basic_data_exploration,
    convert_timestamp_to_datetime,
    create_bar_plots_for_categorical_vs_target,
    create_correlation_heatmap,
    create_3d_scatter_plot,
    create_box_plots_for_categorical_vs_target,
    plot_input_vs_target_line,
    create_violin_plots_for_categorical_vs_target
)

def generate_production_report(model_results, best_model_name, df, X_train, X_test, X_holdout):
    """Generates a comprehensive analysis with ordered model performance"""
    print("\nmodel performance summary:")
    
    # Get unique model performances and sort them
    model_performances = {}
    for model_name, results in model_results.items():
        if isinstance(results, dict) and 'test_rmse' in results:
            if model_name not in model_performances:  # Avoid duplicates
                model_performances[model_name] = {
                    'test_rmse': results['test_rmse'],
                    'holdout_rmse': results.get('holdout_rmse', None)
                }
    
    # Sort by test RMSE
    sorted_models = sorted(model_performances.items(), key=lambda x: x[1]['test_rmse'])
    
    print("\nmodel ranking:")
    for rank, (name, perf) in enumerate(sorted_models, 1):
        print(f"\n{rank}. {name}")
        print(f"   test rmse: {perf['test_rmse']:.4f}")
        if perf['holdout_rmse']:
            print(f"   holdout rmse: {perf['holdout_rmse']:.4f}")
            print(f"   generalization gap: {abs(perf['test_rmse'] - perf['holdout_rmse']):.4f}")

    
##############################################################
if __name__ == "__main__":
##############################################################
    # Read in data 
    ##############################################################
    dataset_id = "ziya07/adas-ev-dataset"
    file = "ADAS_EV_Dataset.csv"

    # Set up argument parsing to allow optional visualization and custom splits
    parser = argparse.ArgumentParser(
        description='EV Data Analysis Pipeline',
        epilog=(
            "\nExample usage:\n"
            "   python main.py                     # Default: 70% train, 20% test, 10% holdout\n"
            "   python main.py --include-visualization   # Include data visualization step\n"
            "   python main.py --train-ratio 0.8 --test-ratio 0.15  # Custom: 80% train, 15% test, 5% holdout\n"
            "   python main.py --help                # Show this help message\n\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--include-visualization', action='store_true',
                        help='Include the data visualization step.')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Proportion of data for training (default: 0.7)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Proportion of data for testing/model selection (default: 0.2)')
    
    args = parser.parse_args()
    
    # Validate split ratios
    if args.train_ratio + args.test_ratio >= 1.0:
        raise ValueError("train_ratio + test_ratio must be less than 1.0 to leave room for holdout set")
    
    holdout_ratio = 1.0 - args.train_ratio - args.test_ratio
    if holdout_ratio < 0.05:
        print("Warning: Holdout set is very small (<5%). Consider adjusting ratios for more reliable validation.")
    
    print(f"data split configuration:")
    print(f"  training: {args.train_ratio:.1%}")
    print(f"  testing: {args.test_ratio:.1%}")
    print(f"  holdout: {holdout_ratio:.1%}")

    # Load the DataFrame using the function
    df = load_kaggle_dataset(dataset_id, file, "dataset")

    if df is not None:
        print("\nrunning pipeline...")
        # Step 1: Perform initial data exploration
        print("\n--- Running Data Exploration ---")
        perform_basic_data_exploration(df)

        # Pre-processing and visualization
        df = convert_timestamp_to_datetime(df)
        
        # Define the target and key features as specified by the user
        target_variable = 'energy_consumption'
        numeric_features = ['speed_kmh', 'acceleration_mps2', 'regen_braking_usage', 'brake_intensity', 'traffic_density']
        categorical_features = ['weather_condition', 'road_type']

        # Step 2: Missing Values
        print("\n--- Running Missing Value Analysis ---")
        find_missing_data(df)
        suggest_imputation_strategy(df)
        print("\n--- Automatically Imputing Missing Values ---")
        df = auto_impute_missing_data(df)

        # Step 3: Outlier Handling for Numeric Features
        print("\n--- Handling Outliers in Numeric Features ---")
        for col in numeric_features:
            df = handle_outliers_iqr_method(df, col)

        df_original = df.copy()

        # Step 4: Feature Engineering
        print("\n--- Feature Engineering ---")
        df = create_combined_ev_features(df)
        df = encode_categorical_features(df, ['weather_condition', 'road_type'])

        # Keep only selected features and target
        selected_features = [
            'instantaneous_power_proxy',
            'total_braking_force',
            'traffic_density'
        ] + [col for col in df.columns if col.startswith('weather_condition_') or col.startswith('road_type_')]

        #New features
        numeric_features = ['instantaneous_power_proxy', 'total_braking_force', 'traffic_density']
        
        # Scale features and transform target
        df_processed, scaler = scale_features_and_transform_target(
            df[selected_features + [target_variable]], 
            selected_features, 
            target_variable
        )

        # Update df for modeling
        df = df_processed

        #Step 5: Create visualizations (optional)
        if args.include_visualization:
            print("\n--- Running Data Visualization ---")
            create_correlation_heatmap(df)
            for x_axis, y_axis in combinations(numeric_features, 2):
                create_3d_scatter_plot(df, x_axis, y_axis, target_variable)
            create_box_plots_for_categorical_vs_target(df_original, categorical_features, target_variable)


        # Step 6: Prepare Train/Test/Holdout Splits
        print("\ndata split preparation")
        X_train, X_test, X_holdout, y_train, y_test, y_holdout = prepare_train_test_holdout_splits(
            df, selected_features, target_variable, 
            train_ratio=args.train_ratio, 
            test_ratio=args.test_ratio
        )
        
        print(f"\ndataset summary:")
        print(f"  features: {len(selected_features)}")
        print(f"  samples: {len(df)}")
        
        # Step 7: Initial Model Comparison (using only train+test data)
        print("\nrunning initial model comparison")
        train_test_df = df.iloc[:len(X_train) + len(X_test)]
        best_initial_model, model_performances = run_model_comparison_pipeline(
            train_test_df, selected_features, target_variable
        )
        
        print(f"\nbest model from initial comparison: {best_initial_model}")
        
        print("\ntraining final models...")
        best_model, best_model_name, holdout_results, model_results = run_final_model_training_with_holdout(
            X_train, y_train, X_test, y_test, X_holdout, y_holdout, selected_features, scaler
        )
        
        generate_production_report(model_results, best_model_name, df, X_train, X_test, X_holdout)
        
        print("\npipeline completed")
        print(f"selected model: {best_model_name}")
        print("models saved in saved_models directory")

    else:
        print("failed to load dataset")
        exit(1)