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


def perform_PCA(filename):
    try:
        #loading the processed datasets and make dummies columns for categorical feature
        data=pd.read_csv(filename, index_col=[0])          
        df = pd.get_dummies(data, drop_first=True, dtype=int)
        print(df)
        

        # component scaling of dataframe
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)
        print(scaled_df)
        
        # Initilization of PCA
        pca=PCA()

        # Fit PCA on your data
        pca.fit(scaled_df)
        print(pca.fit(scaled_df)) 
        print('Original Dimensions: ',scaled_df.shape)
        

       # Individual explained variance
        individual_variances = pca.explained_variance_ratio_
        print('Explained variance ratio:\n', individual_variances)

        # Cumulative explained variance
        cumulative_variances = np.cumsum(pca.explained_variance_ratio_)
        print('Cumulative explained variance:\n', cumulative_variances)

        # Generate a bar plot to visualize individual explained variances
        plt.figure(figsize=(10, 6))
        plt.bar(x=np.arange(1, len(individual_variances) + 1), height=individual_variances, label='Individual Variance', color='blue')

        # Plot cumulative explained variance as a line plot
        plt.plot(np.arange(1, len(cumulative_variances) + 1), cumulative_variances, marker='o', color='red', label='Cumulative Variance')
        plt.axhline(y=0.90, color='g', linestyle='--')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance (%)')
        plt.title('Explained Variance by Principal Component')
        plt.legend()
        plt.grid(True)
        output_name = f"variance_ratio_{os.path.splitext(os.path.basename(filename))[0]}.png"
        plt.savefig(os.path.join('results', output_name))
        # #plt.show()

        #performing PCA with 90%
        pca=PCA(n_components=0.9)
        pca.fit(scaled_df)
        pca_data=pca.transform(scaled_df)
        print(pca_data)
        print('Reduced Dimensions: ',pca_data.shape)

        # converting into dataframe
        data_pca=pd.DataFrame(pca_data)
        cols_pca = [f'PC{i}' for i in range(1, pca.n_components_+1)]
        print(cols_pca)
        df_pca=pd.DataFrame(pca_data, columns=cols_pca, index=data.index)
        print(df_pca)   

        # Plotting PC1 vs PC2
        plt.figure(figsize=(8, 6))
        plt.scatter(df_pca['PC1'], df_pca['PC2'])
        plt.title('PC1 vs PC2')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True)
        output_name = f"PC1vsPC2_{os.path.splitext(os.path.basename(filename))[0]}.png"
        plt.savefig(os.path.join('results', output_name))
        #plt.show()

        # ## Create correlation matrix
        corr_matrix=df_pca.corr()
        # # Obtain the principal components
        principal_components = pca.components_

        # # Calculate the correlation matrix
        # correlation_matrix = np.corrcoef(principal_components)
        # # Convert the correlation matrix to a DataFrame for visualization
        correlation_df = pd.DataFrame(corr_matrix, columns=['PC'+str(i+1) for i in range(len(principal_components))])

        # Visualize the correlation matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".3f")
        plt.title('Correlation Heatmap of Principal Components')
        plt.xlabel('Principal Component')
        plt.ylabel('Principal Component')
        output_name = f"correlation_matrix_{os.path.splitext(os.path.basename(filename))[0]}.png"
        plt.savefig(os.path.join('results', output_name))
        # #plt.show()
        return filename
    
    except Exception as e:
        print(f"An error occurred during preprocessing for dataset: {str(e)}")
        return None
