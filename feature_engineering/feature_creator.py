import pandas as pd
import numpy as np

def create_combined_ev_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new, combined features from existing EV driving dynamics data.
    
    This function is tailored for an energy consumption prediction model.
    It combines related features to provide more meaningful insights
    to the machine learning algorithm.

    Args:
        df (pd.DataFrame): The DataFrame containing the raw EV data.

    Returns:
        pd.DataFrame: The DataFrame with the new combined features added.
    """
    df_copy = df.copy()
    print("\n--- Creating Combined Features for EV Energy Prediction ---")

    # Combine speed and acceleration to create a proxy for instantaneous power.
    # This feature should be a strong predictor of energy consumption.
    # Note: We handle potential negative acceleration (deceleration) as well.
    df_copy['instantaneous_power_proxy'] = df_copy['speed_kmh'] * df_copy['acceleration_mps2']
    print("  'instantaneous_power_proxy' created from speed and acceleration.")
    
    # Combine regenerative braking and brake intensity to represent overall braking force.
    # This can help the model understand energy recovery and loss during deceleration.
    df_copy['total_braking_force'] = df_copy['regen_braking_usage'] + df_copy['brake_intensity']
    print("  'total_braking_force' created from regen_braking_usage and brake_intensity.")
    
    print("  Combined feature engineering complete.")
    
    return df_copy

def encode_categorical_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Performs one-hot encoding on categorical features.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        categorical_cols (list): A list of column names to encode.
        
    Returns:
        pd.DataFrame: The DataFrame with one-hot encoded features.
    """
    print("\n--- Encoding Categorical Features ---")
    # Use pandas get_dummies for one-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
    print("  Categorical features have been one-hot encoded.")
    return df