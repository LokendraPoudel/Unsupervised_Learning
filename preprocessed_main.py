#This code is written by Lokendra Poudel @ Polaron Analytics

import numpy as np
import os
import shutil
import pandas as pd
import math
import sys
from subprocess import call
import time

# get the start time
st = time.time()
import argparse
from src.data.preprocessing import load_data
from src.data.preprocessing import preprocessing_dataset
from src.data.preprocessing import preprocessing_customer


def main(filename):  
    # Load the data
    data = load_data(filename)
    
    # Print the first few rows of the loaded data
    if data is not None:
        print(data.head()) 

    # data processing
    processed_dataset= preprocessing_dataset(filename)
    processed_customer= preprocessing_customer(filename)

def load_data(filename):
    """
    Load data from a CSV file.
    
    Args:from src.data.preprocessing import preprocessing_dataset
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform clustering on a dataset.")
    parser.add_argument("filename", type=str, help="Path to the CSV file containing the dataset.")
    args = parser.parse_args()

    main(args.filename)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')