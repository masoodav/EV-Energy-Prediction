##############################################################
# Import main libraries 
##############################################################
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import argparse

# Import all data preprocessing and feature engineering functions
from data_collection.data_loader import load_kaggle_dataset
from data_preprocessing.data_preprocessor import (
    find_missing_data,
    find_and_replace_non_standard_missing_values,
    impute_missing_data,
    suggest_imputation_strategy,
    auto_impute_missing_data,
    handle_outliers_iqr_method
)
from feature_engineering.feature_creator import (
    create_combined_ev_features,
    encode_categorical_features
)
from cross_validation.train_test_split import prepare_tscv_splits

# New: We now import the main modeling function from the model.py file
from modeling.model import run_model_comparison_pipeline

# Import all data analysis and visualization functions
from exploratory_data_analysis.data_analysis import (
    perform_basic_data_exploration,
    convert_timestamp_to_datetime,
    create_3d_scatter_plot,
    create_box_plots_for_categorical_vs_target
)

##############################################################
if __name__ == "__main__":
##############################################################
    # Read in data 
    ##############################################################
    dataset_id = "ziya07/adas-ev-dataset"
    file = "ADAS_EV_Dataset.csv"

    # Set up argument parsing to allow optional visualization
    parser = argparse.ArgumentParser(
        description='EV Data Analysis Pipeline',
        epilog=(
            "\nExample usage:\n"
            "   python main.py                     # Run all steps except visualization\n"
            "   python main.py --include-visualization   # Include data visualization step\n"
            "   python main.py --help                # Show this help message\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--include-visualization', action='store_true',
                        help='Include the data visualization step.')
    args = parser.parse_args()

    # Load the DataFrame using the function
    df = load_kaggle_dataset(dataset_id, file, "dataset")

    if df is not None:
        # Optional: Clean non-standard missing values before analysis
        df = find_and_replace_non_standard_missing_values(df)

        # Step 1: Perform initial data exploration
        print("\n--- Running Data Exploration ---")
        perform_basic_data_exploration(df)

        # Pre-processing and visualization
        df = convert_timestamp_to_datetime(df)
        
        # Define the target and key features as specified by the user
        target_variable = 'energy_consumption'
        numeric_features = ['speed_kmh', 'acceleration_mps2', 'regen_braking_usage', 'brake_intensity', 'traffic_density']
        categorical_features = ['weather_condition', 'road_type']

        # Step 2: Create visualizations (optional)
        if args.include_visualization:
            print("\n--- Running Data Visualization ---")
            print(f"\n--- Generating all pairwise 3D plots for numeric features vs. {target_variable} ---")
            for x_axis, y_axis in combinations(numeric_features, 2):
                create_3d_scatter_plot(df, x_axis, y_axis, target_variable)
            create_box_plots_for_categorical_vs_target(df, categorical_features, target_variable)

        # Step 3: Missing Values
        print("\n--- Running Missing Value Analysis ---")
        find_missing_data(df)
        suggest_imputation_strategy(df)
        print("\n--- Automatically Imputing Missing Values ---")
        df = auto_impute_missing_data(df)

        # Step 4: Outlier Handling for Numeric Features
        print("\n--- Handling Outliers in Numeric Features ---")
        for col in numeric_features:
            df = handle_outliers_iqr_method(df, col)

        # Step 5: Feature Engineering
        print("\n--- Feature Engineering ---")
        df = create_combined_ev_features(df)
        df = encode_categorical_features(df, ['weather_condition', 'road_type'])

        # Keep only selected features and target
        selected_features = [
            'instantaneous_power_proxy',
            'total_braking_force',
            'traffic_density'
        ] + [col for col in df.columns if col.startswith('weather_condition_') or col.startswith('road_type_')]
        df = df[selected_features + [target_variable]]

        # Step 6: Time-Series Cross-Validation and Model Training
        # We now call the new function in model.py to handle all model training and evaluation
        run_model_comparison_pipeline(df, selected_features, target_variable)
