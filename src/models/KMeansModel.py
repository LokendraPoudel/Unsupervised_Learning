#This code is written by Lokendra Poudel @ Polaron Analytics

import pandas as pd
import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.utils import AutoElbow
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import joblib
import os


def KMeansModel(filename):
    try:
        #loading the processed datasets and make dummies columns for categorical feature
        df=pd.read_csv(filename, index_col=[0])
        df_new= pd.get_dummies(df)
        df_kmeans = pd.get_dummies(df, drop_first=True, dtype=int)
        print(df_kmeans)

        # # standardizing features by removing the mean and scaling to unit variance.
        # scaler = StandardScaler()
        # X = scaler.fit_transform(df_kmeans)
        # print(X)
        
        #Unsupervised machine learning techniques do not require you to split your data into training data and test data 
        #from sklearn.model_selection import train_test_split
        # from sklearn.model_selection import train_test_split
        # X_train, X_test = train_test_split(X, test_size = 0.2, random_state = 0)


        #Selecting optimal number of clustering/Elbow method
        from sklearn.cluster import KMeans
        wcss = []
        for K in range(1, 9):
            kmeans_model = KMeans(n_clusters=K, n_init=10, random_state=42)
            kmeans_model.fit(df_kmeans)
            wcss.append(kmeans_model.inertia_)
        plt.figure(figsize=(8, 5), dpi=100)
        plt.plot(range(1, 9), wcss, color="green", marker="o")
        plt.xlabel("Number of clusters (K)")
        plt.ylabel("WCSS for K")
        output_name = f"ElbowMethod_{os.path.splitext(os.path.basename(filename))[0]}.png"
        plt.savefig(os.path.join('results', output_name))
        #plt.show()

        n =AutoElbow.auto_elbow_search(df_kmeans)
        print("optimized number of clusters:",n)

        #Silhouette score for number of clustering
        range_n_clusters = list(range(2, 9))
        silhouette_scores = []

        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            cluster_labels = clusterer.fit_predict(df_kmeans)

            silhouette_avg = silhouette_score(df_kmeans, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

        plt.figure(figsize=(10,5))
        plt.plot(range_n_clusters, silhouette_scores, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Silhouette score')
        plt.title('Silhouette analysis For KMeans clustering')
        output_name = f"SilhouettsScore_{os.path.splitext(os.path.basename(filename))[0]}.png"
        plt.savefig(os.path.join('results', output_name))
        #plt.show()
        

        # KMean models with k=n
        kmeans_model= KMeans(n_clusters=n, n_init=10, random_state=42)
        kmeans_model.fit(df_kmeans)

        # Save the trained model (optional)
        output_name = f"kmeans_model_{os.path.splitext(os.path.basename(filename))[0]}.pkl"
        joblib.dump(kmeans_model, os.path.join('results', output_name))
        

        # Predict the cluster assignments
        clusters = kmeans_model.fit_predict(df_kmeans)

        # Perform t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(df_kmeans)

        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot original data
        plt.subplot(1, 1, 1)
        ax=plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', marker='.')
        plt.title('Clustered Data (t-SNE Visualization)')
        plt.xlabel('t-SNE Dimension 1')
        plt.colorbar(ax, ticks=np.arange(n_clusters))
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        output_name = f"t-SNE_{os.path.splitext(os.path.basename(filename))[0]}.png"
        plt.savefig(os.path.join('results', output_name))
        #plt.show()

        # Insert the 'Cluster' column with cluster assignments into the DataFrame
        df_kmeans.insert(len(df_kmeans.columns), "Cluster", clusters)

        # Display unique cluster labels
        print(df_kmeans['Cluster'].unique())

        # Print the count of data points in each cluster
        cluster_counts = pd.Series(clusters).value_counts()
        print(cluster_counts)

        #clustering distribution
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        # Plotting cluster vs count
        ax = sns.countplot(x=clusters, ax=axes[0])
        ax.bar_label(ax.containers[0], fontsize=12)
        axes[0].set_title('Cluster Distribution')
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Count')
        # Annotate each bar with its count value
        # for container in ax.containers:
        #     ax.bar_label(container,fmt="%.2f", fontsize=12)

        # Plotting cluster vs percentage
        ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values / len(clusters) * 100, ax=axes[1])

        ax.set_title('Cluster Distribution (in Percentage)')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Percentage (%)')

        # Annotate each bar with its percentage value
        for container in ax.containers:
            ax.bar_label(container,fmt="%.2f", fontsize=12)

        plt.tight_layout()
        output_name = f"clusters_distribution_{os.path.splitext(os.path.basename(filename))[0]}.png"
        plt.savefig(os.path.join('results', output_name))
        #plt.show()
        

        #Prediction
        preds = kmeans_model.labels_
        df_kmeans = pd.DataFrame(df)
        df_kmeans['Clusters'] = preds
        #print(df_kmeans)    

        #save clustering dataset
        output_name = f"clustering_{os.path.splitext(os.path.basename(filename))[0]}.csv"
        df_kmeans.to_csv(os.path.join('data', output_name))
        #df_kmeans.to_csv ("data/clustering_dataset.csv")
        return filename
    except Exception as e:
        print(f"An error occurred during preprocessing for dataset: {str(e)}")
        return None
    return










