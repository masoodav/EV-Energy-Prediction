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
    find_and_replace_non_standard_missing_values,
    impute_missing_data,
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
    tune_lightgbm_model,
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

    # Set up argument parsing to allow optional visualization and custom splits
    parser = argparse.ArgumentParser(
        description='EV Data Analysis Pipeline',
        epilog=(
            "\nExample usage:\n"
            "   python main.py                     # Default: 70% train, 20% test, 10% holdout\n"
            "   python main.py --include-visualization   # Include data visualization step\n"
            "   python main.py --train-ratio 0.8 --test-ratio 0.15  # Custom: 80% train, 15% test, 5% holdout\n"
            "   python main.py --help                # Show this help message\n\n"
            "The holdout dataset is NEVER used during training, hyperparameter tuning,\n"
            "or model selection. It's only used at the very end for final validation.\n"
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
    
    print(f"Data split configuration:")
    print(f"  Training: {args.train_ratio:.1%} - for model training")
    print(f"  Testing: {args.test_ratio:.1%} - for model selection and hyperparameter tuning")  
    print(f"  Holdout: {holdout_ratio:.1%} - RESERVED for final validation (never seen during training)")

    # Load the DataFrame using the function
    df = load_kaggle_dataset(dataset_id, file, "dataset")

    if df is not None:

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
        
        # Scale features and transform target
        df_processed, scaler = scale_features_and_transform_target(
            df[selected_features + [target_variable]], 
            selected_features, 
            target_variable
        )

        # Update df for modeling
        df = df_processed

        # Step 6: Prepare Train/Test/Holdout Splits
        print("\n" + "="*80)
        print("PREPARING TRAIN/TEST/HOLDOUT SPLITS")
        print("="*80)
        print(" IMPORTANT: The holdout dataset will be completely reserved!")
        
        # Create train/test/holdout splits maintaining chronological order
        X_train, X_test, X_holdout, y_train, y_test, y_holdout = prepare_train_test_holdout_splits(
            df, selected_features, target_variable, 
            train_ratio=args.train_ratio, 
            test_ratio=args.test_ratio
        )
        
        print(f"\nDataset information:")
        print(f"  Features: {selected_features}")
        print(f"  Target variable: {target_variable}")
        print(f"  Total samples: {len(df)}")
        
        # Step 7: Initial Model Comparison (using only train+test data)
        print("\n" + "="*80)
        print("INITIAL MODEL COMPARISON")
        print("="*80)
        print(" Running cross-validation using train+test portion of data...")
        
        # Create a subset of data that excludes holdout for CV
        train_test_df = df.iloc[:len(X_train) + len(X_test)]
        run_model_comparison_pipeline(train_test_df, selected_features, target_variable)
        
        # Step 8: Final Model Training and Selection (using only train+test data)
        print("\n" + "="*80)
        print("FINAL MODEL TRAINING AND SELECTION")
        print("="*80)
        print(" Training and selecting best model using train+test data...")
        
        # Run the comprehensive model training pipeline with holdout validation
        best_model, best_model_name, holdout_results, model_results = run_final_model_training_with_holdout(
            X_train, y_train, X_test, y_test, X_holdout, y_holdout, selected_features
        )
        
        # Step 9: Comprehensive Analysis and Production Recommendations
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS AND PRODUCTION RECOMMENDATIONS")
        print("="*80)
        
        # Extract results for analysis
        lgb_test_rmse = model_results['lgb_test_rmse']
        lgb_holdout_rmse = model_results['lgb_holdout_rmse']
        stack_test_rmse = model_results['stack_test_rmse'] 
        stack_holdout_rmse = model_results['stack_holdout_rmse']
        
        # Model selection summary
        print(f"\n SELECTED MODEL: {best_model_name}")
        print(f"   Selected based on test set performance during training")
        
        # Holdout validation summary
        print(f"\n HOLDOUT VALIDATION RESULTS (Real-world Performance Estimate):")
        print(f"   LightGBM Holdout RMSE:      {lgb_holdout_rmse:.4f}")
        print(f"   Stacking Holdout RMSE:      {stack_holdout_rmse:.4f}")
        
        # Generalization analysis
        lgb_generalization_gap = abs(lgb_test_rmse - lgb_holdout_rmse)
        stack_generalization_gap = abs(stack_test_rmse - stack_holdout_rmse)
        
        print(f"\n GENERALIZATION ANALYSIS:")
        print(f"   LightGBM:")
        print(f"     Test RMSE: {lgb_test_rmse:.4f}")
        print(f"     Holdout RMSE: {lgb_holdout_rmse:.4f}")
        print(f"     Gap: {lgb_generalization_gap:.4f}")
        
        print(f"   Stacking Ensemble:")
        print(f"     Test RMSE: {stack_test_rmse:.4f}")
        print(f"     Holdout RMSE: {stack_holdout_rmse:.4f}")
        print(f"     Gap: {stack_generalization_gap:.4f}")
        
        # Best model analysis
        if best_model_name == "LightGBM":
            final_holdout_rmse = lgb_holdout_rmse
            generalization_gap = lgb_generalization_gap
        else:
            final_holdout_rmse = stack_holdout_rmse
            generalization_gap = stack_generalization_gap
        
        # Performance assessment
        print(f"\n PERFORMANCE ASSESSMENT:")
        if generalization_gap < 0.01:
            print(f"    EXCELLENT: Very low generalization gap ({generalization_gap:.4f})")
            reliability = "High"
        elif generalization_gap < 0.03:
            print(f"    GOOD: Acceptable generalization gap ({generalization_gap:.4f})")
            reliability = "Good"
        elif generalization_gap < 0.05:
            print(f"     FAIR: Moderate generalization gap ({generalization_gap:.4f})")
            reliability = "Fair"
        else:
            print(f"    POOR: High generalization gap ({generalization_gap:.4f})")
            reliability = "Poor"
        
        # Data split assessment
        print(f"\n DATA SPLIT ASSESSMENT:")
        print(f"   Training samples: {len(X_train)} ({len(X_train)/len(df):.1%})")
        print(f"   Test samples: {len(X_test)} ({len(X_test)/len(df):.1%})")
        print(f"   Holdout samples: {len(X_holdout)} ({len(X_holdout)/len(df):.1%})")
        
        if len(X_holdout) < 50:
            print(f"     Very small holdout set - consider increasing holdout ratio")
        elif len(X_holdout) < 100:
            print(f"     Small holdout set - results may be less reliable")
        else:
            print(f"    Adequate holdout set size for reliable validation")
        
        # Production deployment recommendations
        print(f"\n PRODUCTION DEPLOYMENT RECOMMENDATIONS:")
        print(f"   1.  Deploy: {best_model_name}")
        print(f"   2.  Expected Performance: {final_holdout_rmse:.4f} RMSE")
        print(f"   3.  Model Reliability: {reliability}")
        print(f"   4.  Performance Monitoring:")
        print(f"      • Set performance alert if RMSE exceeds {final_holdout_rmse + 0.02:.4f}")
        print(f"      • Consider retraining if RMSE exceeds {final_holdout_rmse + 0.05:.4f}")
        print(f"   5.  A/B Testing:")
        print(f"      • Both models saved for A/B testing")
        print(f"      • Compare live performance against holdout validation")
        print(f"   6.  Model Validation:")
        print(f"      • Holdout RMSE represents true unseen data performance")
        print(f"      • Use this as baseline for production monitoring")
        
        # Risk assessment
        print(f"\n  RISK ASSESSMENT:")
        if generalization_gap > 0.03:
            print(f"   • HIGH: Significant performance drop on unseen data")
            print(f"   • Recommend additional validation before production")
        elif generalization_gap > 0.01:
            print(f"   • MEDIUM: Some performance drop expected")
            print(f"   • Monitor closely in production")
        else:
            print(f"   • LOW: Performance consistent across datasets")
            print(f"   • Ready for production deployment")
        
        # Final summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f" Production Model: {best_model_name}")
        print(f" Test Set RMSE: {lgb_test_rmse if best_model_name == 'LightGBM' else stack_test_rmse:.4f}")
        print(f" Holdout RMSE (Real-world estimate): {final_holdout_rmse:.4f}")
        print(f" Generalization Gap: {generalization_gap:.4f}")
        print(f" All models and results saved in 'saved_models' directory")
        print(f" Holdout validation completed - model ready for production!")
        
        # Test model loading
        print(f"\n--- Testing Model Loading for Production Deployment ---")
        try:
            model_files = model_results['model_files']
            # Load the selected best model
            if best_model_name == "LightGBM":
                loaded_model_data = load_model_for_prediction(model_files['lgb_file'])
            else:
                loaded_model_data = load_model_for_prediction(model_files['stack_file'])
            
            # Make a test prediction on a sample
            if len(X_holdout) > 0:
                sample_X = X_holdout.iloc[:1]  # Get first holdout sample
                sample_y_actual = y_holdout.iloc[0]
                
                # Convert to numpy if needed for ensemble models
                if hasattr(sample_X, 'values'):
                    sample_X_array = sample_X.values
                else:
                    sample_X_array = sample_X
                    
                sample_prediction = loaded_model_data['model'].predict(sample_X_array)[0]
                prediction_error = abs(sample_prediction - sample_y_actual)
                
                print(f" Model loading test successful!")
                print(f"   Sample prediction: {sample_prediction:.4f}")
                print(f"   Actual value: {sample_y_actual:.4f}")
                print(f"   Prediction error: {prediction_error:.4f}")
                print(" Model ready for production deployment!")
            else:
                print(" Model loading test successful - ready for production!")
                
        except Exception as e:
            print(f" Model loading test failed: {e}")
            print("Please check saved model files before production deployment.")
            
    else:
        print("Failed to load dataset. Please check the dataset_id and file name.")
        exit(1)