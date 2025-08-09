import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

def find_missing_data(df: pd.DataFrame):
    """
    Identifies columns with missing values and reports their count and percentage.

    Args:
        df: The DataFrame to analyze for missing values.
    """
    print("\n--- Missing Value Analysis ---")
    
    # Calculate the total number of rows
    total_rows = len(df)
    
    # Get a Series of columns with any missing values
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if missing_data.empty:
        print("No missing values found in the dataset.")
        return
    
    # Create a DataFrame to present the results
    missing_df = pd.DataFrame({
        'Total Missing': missing_data,
        'Percentage': (missing_data / total_rows) * 100
    })
    
    print("Columns with missing values and their details:")
    print(missing_df)

    # Provide guidance on potential imputation methods based on missing percentage
    print("\n--- Imputation Method Suggestions ---")
    for column, row in missing_df.iterrows():
        print(f"Column: '{column}'")
        print(f"  Data Type: {df[column].dtype}")
        if row['Percentage'] < 5:
            print("  Suggestion: Missing values are low. A simple method like mean/median/mode imputation is likely sufficient.")
        elif row['Percentage'] < 20:
            print("  Suggestion: Missing values are moderate. Consider more advanced methods like K-Nearest Neighbors (KNN) imputation or domain-specific imputation.")
        else:
            print("  Suggestion: Missing values are high. Advanced imputation or dropping the column/rows may be necessary. A careful analysis is required.")

def find_and_replace_non_standard_missing_values(df):
    """
    Identifies and replaces non-standard missing values with NaN.

    This function looks for common non-standard missing value representations
    and converts them to numpy.nan, making them detectable by standard
    pandas missing value functions.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with non-standard missing values replaced.
    """
    # Define a list of common non-standard missing values to search for
    non_standard_values = ["N/A", "n/a", "na", "--"]
    # Replace empty strings and spaces only in object (string) columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(["", " "], np.nan)
        df[col] = df[col].replace(non_standard_values, np.nan)
    # Replace numeric placeholders only in numeric columns
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].replace([-999, -9999], np.nan)

    print("Non-standard missing values replaced with NaN.")
    print("New missing value count:")
    print(df.isnull().sum())
    return df

def impute_missing_data(df, strategy="mean", variable=None):
    """
    Impute missing data using various strategies.

    Args:
        df (pd.DataFrame): DataFrame to impute.
        strategy (str): One of 'drop', 'mean', 'median', 'mode', 'iterative', 'knn'.
        variable (str, optional): Column name for single-column imputation.

    Returns:
        pd.DataFrame: Imputed DataFrame.
    """
    if strategy == "drop":
        return df.dropna()
    elif strategy == "mean" and variable:
        df[variable] = df[variable].fillna(np.mean(df[variable]))
        return df
    elif strategy == "median" and variable:
        df[variable] = df[variable].fillna(np.median(df[variable]))
        return df
    elif strategy == "mode" and variable:
        df[variable] = df[variable].fillna(df[variable].mode()[0])
        return df
    elif strategy == "iterative":
        imp = IterativeImputer(max_iter=10, random_state=0)
        imputed = imp.fit_transform(df)
        return pd.DataFrame(imputed, columns=df.columns)
    elif strategy == "knn":
        knn_imp = KNNImputer(n_neighbors=5, weights="uniform")
        imputed = knn_imp.fit_transform(df)
        return pd.DataFrame(imputed, columns=df.columns)
    else:
        raise ValueError("Invalid strategy or missing variable name for single-column imputation.")

def suggest_imputation_strategy(df):
    """
    Suggests an imputation strategy for each column with missing values.

    Considers missing percentage and data type.
    """
    total_rows = len(df)
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    print("\n--- Imputation Strategy Suggestions ---")
    for column in missing_data.index:
        pct = (missing_data[column] / total_rows) * 100
        dtype = df[column].dtype
        print(f"Column: '{column}' | Missing: {missing_data[column]} ({pct:.2f}%) | Type: {dtype}")
        if pct < 5:
            if np.issubdtype(dtype, np.number):
                print("  → Use mean or median imputation.")
            else:
                print("  → Use mode imputation.")
        elif pct < 20:
            print("  → Consider KNN or Iterative Imputer for more robust results.")
        else:
            print("  → Consider dropping column/rows or advanced imputation. Review carefully.")

def auto_impute_missing_data(df, moderate_method="knn", high_threshold=20):
    """
    Automatically imputes missing data for each column based on suggested strategy.

    Args:
        df (pd.DataFrame): DataFrame to impute.
        moderate_method (str): 'knn' or 'iterative' for moderate missingness.
        high_threshold (float): Percentage threshold to consider missingness as high.

    Returns:
        pd.DataFrame: Imputed DataFrame.
    """
    total_rows = len(df)
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    df_imputed = df.copy()

    low_threshold = 5

    for column in missing_data.index:
        pct = (missing_data[column] / total_rows) * 100
        dtype = df[column].dtype

        if pct < low_threshold:
            if np.issubdtype(dtype, np.number):
                df_imputed[column] = df_imputed[column].fillna(np.mean(df_imputed[column]))
            else:
                df_imputed[column] = df_imputed[column].fillna(df_imputed[column].mode()[0])
        elif pct < high_threshold:
            # For moderate missingness, use KNN or Iterative Imputer for numeric columns
            if np.issubdtype(dtype, np.number):
                if moderate_method == "knn":
                    knn_imp = KNNImputer(n_neighbors=5, weights="uniform")
                    df_imputed[[column]] = knn_imp.fit_transform(df_imputed[[column]])
                elif moderate_method == "iterative":
                    imp = IterativeImputer(max_iter=10, random_state=0)
                    df_imputed[[column]] = imp.fit_transform(df_imputed[[column]])
            else:
                # For categorical, use mode
                df_imputed[column] = df_imputed[column].fillna(df_imputed[column].mode()[0])
        else:
            # High missingness: drop column
            print(f"Column '{column}' has high missingness ({pct:.2f}%). Dropping column.")
            df_imputed = df_imputed.drop(columns=[column])

    return df_imputed

# Usage guidance:
"""
How to decide which imputation strategy to use:
- For <5% missing: Use mean/median (numeric) or mode (categorical).
- For 5-20% missing: Use KNNImputer or IterativeImputer for numeric columns.
- For >20% missing: Consider dropping the column/rows, or use domain-specific methods.
- Always check the data type: mean/median for numeric, mode for categorical.
Call suggest_imputation_strategy(df) to get recommendations for your dataset.
"""

def handle_outliers_iqr_method(df: pd.DataFrame, column: str, strategy: str = 'cap') -> pd.DataFrame:
    """
    Handles outliers in a specified column using the IQR method.

    The IQR method is robust and suitable for both normally and non-normally
    distributed data.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        column (str): The name of the column to handle outliers in.
        strategy (str): 'cap' to cap outliers at the bounds, or 'remove' to
                        delete rows containing outliers.

    Returns:
        pd.DataFrame: The DataFrame with outliers handled.
    """
    df_copy = df.copy()
    
    # Calculate Q1, Q3, and the IQR
    Q1 = df_copy[column].quantile(0.25)
    Q3 = df_copy[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the upper and lower bounds for outliers
    upper_bound = Q3 + (1.5 * IQR)
    lower_bound = Q1 - (1.5 * IQR)
    
    outliers_mask = (df_copy[column] < lower_bound) | (df_copy[column] > upper_bound)
    num_outliers = outliers_mask.sum()

    if num_outliers == 0:
        print(f"\nNo IQR outliers found in '{column}'.")
        return df_copy

    print(f"\n--- Handling {num_outliers} IQR outliers in '{column}' ---")

    if strategy == 'cap':
        df_copy[column] = np.where(df_copy[column] > upper_bound, upper_bound, df_copy[column])
        df_copy[column] = np.where(df_copy[column] < lower_bound, lower_bound, df_copy[column])
        print(f"  Outliers in '{column}' have been capped at the IQR bounds.")
    elif strategy == 'remove':
        initial_rows = len(df_copy)
        df_copy = df_copy[~outliers_mask].reset_index(drop=True)
        print(f"  Removed {initial_rows - len(df_copy)} rows with outliers in '{column}'.")
    else:
        print("  Invalid strategy. Choose 'cap' or 'remove'. No action taken.")
    
    return df_copy

def scale_features_and_transform_target(df, features, target_variable):
    """
    Scales features using StandardScaler and applies log transform to target.
    """
    df_scaled = df.copy()
    
    # Scale features
    scaler = StandardScaler()
    df_scaled[features] = scaler.fit_transform(df[features])
    
    # Log transform the target (adding small constant to avoid log(0))
    df_scaled[target_variable] = np.log1p(df_scaled[target_variable])
    
    print("\n--- Feature Scaling and Target Transformation ---")
    print("Features standardized to zero mean and unit variance")
    print(f"Target '{target_variable}' log-transformed")
    
    return df_scaled, scaler