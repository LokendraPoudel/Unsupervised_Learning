import pandas as pd
import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filename):
    """
    Load data from a CSV file.
    
    Args:
    - data_path (str): Path to the CSV file.
    
    Returns:
    - DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print(f"Error: File '{data}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {str(e)}")
        return None

def preprocessing_dataset(filename):
    try:
        #loading the raw datasets
        df=pd.read_csv(filename)
        df.sample(5, random_state=44)
        df.info()
        df=df.drop(['ID', 'Segmentation'], axis="columns")
        #df.info()
        
        # dropping notvalued rows
        df=df.dropna()
        
        # resetting indexing and dropping index column
        df = df.reset_index()
        df = df.drop("index", axis="columns")
        df.info()

        #save processed dataset
        df.to_csv ("data/processed_dataset.csv")
        return filename
    except Exception as e:
        print(f"An error occurred during preprocessing for dataset: {str(e)}")
        return None

def preprocessing_customer(filename):
    
    try:
        #loading the raw datasets
        df=pd.read_csv(filename)
        df.sample(5, random_state=44)
        df = df.drop(["ID", "Dt_Customer", "Z_CostContact", "Z_Revenue"], axis="columns")
        new_column_name='Older'
        df[new_column_name]= 2024-df['Year_Birth']
        df = df.drop(["Year_Birth"], axis="columns")
        
        # dropping notvalued rows
        df=df.dropna()
        
        # resetting indexing and dropping index column
        df = df.reset_index()
        df = df.drop("index", axis="columns")
        df.info()
        df.head()

        #save processed dataset
        df.to_csv ("data/processed_customer_segmentation.csv")
        return filename
    except Exception as e:
        print(f"An error occurred during preprocessing for customer_segmentation: {str(e)}")
        return None
