##############################################################
# Import main libraries 
##############################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_basic_data_exploration(df):
    """
    Performs basic data exploration on a given pandas DataFrame.

    This function prints:
    - The column names.
    - The shape (number of rows and columns).
    - Descriptive statistics for numeric columns.
    - Descriptive statistics for categorical columns.
    
    Args:
        df (pd.DataFrame): The DataFrame to explore.
    """
    if df is not None:
        print("\n--- Basic Data Exploration ---")
        
        # Show columns.
        print("\nDataFrame Columns:")
        print(df.columns)

        # Show number of rows and columns.
        print("\nDataFrame Shape (Rows, Columns):")
        print(df.shape)

        # Show descriptive statistics for numeric variables.
        print("\nDescriptive Statistics for Numeric Variables:")
        print(df.describe())

        # Show descriptive statistics for categorical variables.
        # The include='object' argument is used for non-numeric columns.
        print("\nDescriptive Statistics for Categorical Variables:")
        print(df.describe(include='object'))
    else:
        print("\nFailed to load the dataset.")

def convert_timestamp_to_datetime(df):
    """
    Converts the 'timestamp' column of a DataFrame to datetime objects.

    Args:
        df (pd.DataFrame): The DataFrame with a 'timestamp' column.

    Returns:
        pd.DataFrame: The DataFrame with the 'timestamp' column converted.
    """
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print("\n'timestamp' column successfully converted to datetime.")
    return df

def create_histograms_for_numeric_data(df):
    """
    Generates histograms for all numeric columns in the DataFrame, excluding
    the timestamp.

    Args:
        df (pd.DataFrame): The DataFrame to visualize.
    """
    print("\n--- Generating Histograms for Numeric Data ---")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Exclude the timestamp column for histogram generation
    if 'timestamp' in df.columns:
        numeric_df = numeric_df.drop('timestamp', axis=1, errors='ignore')

    if not numeric_df.empty:
        numeric_df.hist(figsize=(15, 15), bins=30, edgecolor='black')
        plt.suptitle("Histograms of Numeric Variables", y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
    else:
        print("No numeric data to plot.")

def create_bar_plots_for_categorical_data(df):
    """
    Generates bar plots for all categorical columns in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to visualize.
    """
    print("\n--- Generating Bar Plots for Categorical Data ---")
    categorical_df = df.select_dtypes(include=['object', 'category'])
    
    if not categorical_df.empty:
        for column in categorical_df.columns:
            plt.figure(figsize=(8, 5))
            df[column].value_counts().plot(kind='bar', color='skyblue')
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    else:
        print("No categorical data to plot.")

def create_correlation_heatmap(df):
    """
    Generates a correlation heatmap for the numeric columns in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to visualize.
    """
    print("\n--- Generating Correlation Heatmap ---")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if not numeric_df.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix of Numeric Variables')
        plt.tight_layout()
        plt.show()
    else:
        print("No numeric data to calculate correlation.")

def create_scatter_plots_for_targets(df):
    """
    Generates scatter plots to visualize the relationship between key
    features and the target variables (energy consumption and battery level).
    
    Args:
        df (pd.DataFrame): The DataFrame to visualize.
    """
    print("\n--- Generating Scatter Plots for Target Variables ---")
    
    target_variables = ['energy_consumption', 'battery_level']
    key_features = ['speed_kmh', 'acceleration_mps2', 'regen_braking_usage', 
                    'brake_intensity','traffic_density','weather_condition', 'road_type']
    
    for target in target_variables:
        for feature in key_features:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[feature], y=df[target], alpha=0.6, color='darkblue')
            plt.title(f'{feature} vs. {target}', fontsize=16)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel(target, fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def create_filtered_scatter_plots_for_targets(df, filter_column, filter_value):
    """
    Generates scatter plots for a filtered subset of the data.
    
    Args:
        df (pd.DataFrame): The DataFrame to visualize.
        filter_column (str): The column name to filter by (e.g., 'road_type').
        filter_value (str): The value to filter for (e.g., 'highway').
    """
    print(f"\n--- Generating Scatter Plots for '{filter_column}' == '{filter_value}' ---")
    
    # Filter the DataFrame based on the provided column and value
    filtered_df = df[df[filter_column] == filter_value]
    
    if filtered_df.empty:
        print(f"No data found for {filter_column} == {filter_value}. Skipping plots.")
        return

    target_variables = ['energy_consumption']
    key_features = ['speed_kmh', 'acceleration_mps2', 'regen_braking_usage', 'brake_intensity']
    
    for target in target_variables:
        for feature in key_features:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=filtered_df[feature], y=filtered_df[target], alpha=0.6, color='darkgreen')
            plt.title(f'{feature} vs. {target} (Filtered by {filter_value})', fontsize=16)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel(target, fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def create_3d_scatter_plot(df, x_axis, y_axis, z_axis, sample_size=100, colormap='viridis'):
    """
    Generates a parameterized 3D scatter plot to visualize the interaction
    between three variables.

    Args:
        df (pd.DataFrame): The DataFrame to visualize.
        x_axis (str): The column name for the x-axis.
        y_axis (str): The column name for the y-axis.
        z_axis (str): The column name for the z-axis (the target variable).
        sample_size (int): The number of data points to sample for clarity.
        colormap (str): The colormap to use for coloring the data points.
    """
    print(f"\n--- Generating 3D Scatter Plot: {x_axis}, {y_axis} vs. {z_axis} ---")
    
    # Prepare the data, dropping NaN values to ensure a clean plot
    plot_df = df.dropna(subset=[x_axis, y_axis, z_axis])
    
    # Sample a limited number of data points for better clarity
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(n=sample_size, random_state=42)

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the data points, coloring them by the z_axis value
    scatter_plot = ax.scatter(plot_df[x_axis], plot_df[y_axis], plot_df[z_axis], 
                              c=plot_df[z_axis], cmap=colormap, s=50)
    
    # Add a color bar to show the scale of the z_axis variable
    fig.colorbar(scatter_plot, ax=ax, shrink=0.5, aspect=5, label=f'{z_axis}')
    
    # Set labels and title
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)
    ax.set_title(f'3D Scatter Plot: {x_axis} and {y_axis} vs. {z_axis} (Color-Coded)')

    plt.tight_layout()
    plt.show()


def create_box_plots_for_categorical_vs_target(df, categorical_features, target_variable):
    """
    Generates box plots to visualize the relationship between categorical
    features and a target variable.
    
    Args:
        df (pd.DataFrame): The DataFrame to visualize.
        categorical_features (list): List of categorical column names.
        target_variable (str): The target variable column name.
    """
    print(f"\n--- Generating Box Plots for Categorical Features vs. {target_variable} ---")
    
    for feature in categorical_features:
        if feature in df.columns and target_variable in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature], y=df[target_variable])
            plt.title(f'Distribution of {target_variable} by {feature}')
            plt.xlabel(feature)
            plt.ylabel(target_variable)
            plt.tight_layout()
            plt.show()