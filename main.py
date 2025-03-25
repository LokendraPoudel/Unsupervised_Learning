#This code is written by 'Lokendra Poudel' @ Polaron Analytics

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
from src.utils.FeaturesAnalysis import FeatureAnalysis
from src.models.KMeansModel import KMeansModel
from src.models.Hierarchical_clustering import HierarchicalClustering
from src.models.PCA import perform_PCA


def main(filename): 
    # Feature Analysis
    #feature_analysis_dataset= FeatureAnalysis(filename)
    #KMeans Model
    #KMeans_Model= KMeansModel(filename)
    #Hierarchical Clustering
    #Hierarchical_Clustering_dataset= HierarchicalClustering(filename)
    # Prinicipal component analysis
    pca_dataset=perform_PCA(filename)



def load_data(filename):
    """
    Load data from a CSV file.
    
    Args:
    - data_path (str): Path to the CSV file.
    
    Returns:
    - DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(filename, index_col=[0])
        # Print the first few rows of the loaded data
        if data is not None:
            print(data.head())
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