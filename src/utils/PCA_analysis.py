#This code is written by Lokendra Poudel @ Polaron Analytics
import pandas as pd
import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
import os

def PCA_Analysis_dataset(filename):
    try:
        #loading the processed datasets and make dummies columns for categorical feature
        data=pd.read_csv(filename, index_col=[0])
        # Perform label encoding for 'Marital_Status'
        feature=data['Var_1']
        feature=pd.DataFrame(feature)
        # dataframe for PCA
        df=data.drop(['Var_1'], axis=1)        
        df = pd.get_dummies(df, drop_first=True, dtype=int)
        print(df) 
                
        # component scaling of dataframe
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)
        print(scaled_df)

        # Perform t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(scaled_df)
        print(X_tsne)

        # Plotting
        plt.figure(figsize=(8, 6))
 
        # Encode 'Var_1' column into numerical values
        label_encoder = LabelEncoder()
        feature['Var_1_Encoded'] = label_encoder.fit_transform(feature['Var_1'])
        
        plt.subplot(1, 1, 1)
        ax=plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=feature['Var_1_Encoded'], cmap='viridis')
        plt.title(' Data (t-SNE Visualization)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        cbar = plt.colorbar(ax, ticks=np.arange(len(label_encoder.classes_)), label='Var_1 Encoded')
        cbar.set_ticklabels(label_encoder.classes_)  # Set tick labels to categorical values
        plt.tight_layout()
        output_name = f"PCA data t-SNE_{os.path.splitext(os.path.basename(filename))[0]}.png"
        plt.savefig(os.path.join('results', output_name))
        #plt.show()
        return filename
        
    except Exception as e:
        print(f"An error occurred during preprocessing for dataset: {str(e)}")
        return None
    return

def PCA_Analysis_customer(filename):
    try:
        #loading the processed datasets and make dummies columns for categorical feature
        data=pd.read_csv(filename, index_col=[0])

        # Perform label encoding for 'Marital_Status'
        feature=data['Marital_Status']
        feature=pd.DataFrame(feature)
        print(feature)
        df=data.drop(['Marital_Status'], axis=1)        
        df = pd.get_dummies(df, drop_first=True, dtype=int)
        print(df)
        

        # component scaling of dataframe
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)
        print(scaled_df)

        # Perform t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(scaled_df)
        print(X_tsne)

        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot original data
        # Encode 'Marital_Status' column into numerical values
        label_encoder = LabelEncoder()
        feature['Marital_Status_Encoded'] = label_encoder.fit_transform(feature['Marital_Status'])
        
        plt.subplot(1, 1, 1)
        ax=plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=feature['Marital_Status_Encoded'], cmap='viridis')
        plt.title(' Data (t-SNE Visualization)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        cbar = plt.colorbar(ax, ticks=np.arange(len(label_encoder.classes_)), label='Marital Status Encoded')
        cbar.set_ticklabels(label_encoder.classes_)  # Set tick labels to categorical values
        plt.tight_layout()
        output_name = f"PCA data t-SNE_{os.path.splitext(os.path.basename(filename))[0]}.png"
        plt.savefig(os.path.join('results', output_name))
        #plt.show()
        return filename
        
    
    
    except Exception as e:
        print(f"An error occurred during preprocessing for dataset: {str(e)}")
        return None
    
    return