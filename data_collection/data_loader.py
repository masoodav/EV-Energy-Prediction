# data_collection/data_loader.py

import os
import pandas as pd
import kagglehub

def load_kaggle_dataset(dataset_id: str, file_name: str, destination_dir: str = None) -> pd.DataFrame or None:
    """
    Downloads a specific file from a Kaggle dataset and loads it into a pandas DataFrame.

    Args:
        dataset_id (str): The identifier for the Kaggle dataset (e.g., "ziya07/adas-ev-dataset").
                          To download the latest version, do not specify a version number.
                          For example, use "ziya07/adas-ev-dataset" instead of "ziya07/adas-ev-dataset/versions/1".
        file_name (str): The name of the specific file to load from the dataset.
        destination_dir (str, optional): The directory where the dataset should be downloaded.
                                         If None, the default Kaggle cache directory is used.

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing the data, or None if an error occurs.
    """
    try:
        # If a destination directory is specified, set the KAGGLEHUB_CACHE environment variable.
        # This is the recommended way to change the download location for kagglehub.
        if destination_dir:
            os.environ['KAGGLEHUB_CACHE'] = os.path.abspath(destination_dir)
            # Create the destination directory if it does not exist
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)

        # Use the kagglehub library to download the dataset to a local path.
        # This requires you to have the Kaggle API token set up.
        # See https://www.kaggle.com/docs/api for details on authentication.
        
        # Download the dataset. This will download the latest version unless a specific version is provided.
        download_path = kagglehub.dataset_download(dataset_id)
        
        # Construct the full path to the specific file
        file_path = os.path.join(download_path, file_name)
        
        # Print the download path for easy debugging and file verification.
        print(f"Dataset downloaded to: {download_path}")

        if os.path.exists(file_path):
            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            print(f"Successfully downloaded and loaded '{file_name}' from '{dataset_id}'.")
            return df
        else:
            # This block is useful for when the file name provided is incorrect.
            # You can check the downloaded directory for the correct file name.
            print(f"Error: The file '{file_name}' was not found in the downloaded dataset at '{file_path}'.")
            print("Please verify the file name. You can check the contents of the downloaded folder.")
            return None

    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        print("This could be due to a network issue, an incorrect dataset ID, or authentication problems.")
        return None
    finally:
        # Always unset the environment variable to avoid side effects in other parts of the program.
        if destination_dir:
            del os.environ['KAGGLEHUB_CACHE']

# Example usage:
# df = load_kaggle_dataset(dataset_id="ziya07/adas-ev-dataset", file_name="ADAS EV Dataset.csv", destination_dir="my_datasets")
