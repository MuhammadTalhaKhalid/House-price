import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def load_housing_data():
    """
    Downloads the housing prices dataset from Kaggle and loads it into a Pandas DataFrame.
    Returns:
        pd.DataFrame: The housing dataset.
    """

    # Set Kaggle credentials (replace with your own)
    os.environ['KAGGLE_USERNAME'] = "talhakhalid1996"
    os.environ['KAGGLE_KEY'] = "84881a21d6519ce7088cdb6b36cf4517"

    # Authenticate with Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    api.dataset_download_files('yasserh/housing-prices-dataset', path='.', unzip=True)

    # Load the CSV file
    df = pd.read_csv('Housing.csv')
    return df

if __name__ == '__main__':
    df = load_housing_data()
    print("Data loaded successfully!")
    print(df.head())