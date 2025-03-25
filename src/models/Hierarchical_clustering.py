#This code is written by Lokendra Poudel @ Polaron Analytics

import pandas as pd
import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from src.utils import AutoElbow
import os 

def HierarchicalClustering(filename):
    try:
        #loading the processed datasets and make dummies columns for categorical feature
        df=pd.read_csv(filename, index_col=[0])
        df_dum = pd.get_dummies(df, drop_first=True, dtype=int)
        print(df_dum)
        # Hierarchical clustering
        hc = AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(df_dum)

        # Plot the dendrogram
        plt.figure(figsize=(25, 20))
        dendrogram = sch.dendrogram(sch.linkage(df_dum, method='ward'))
        plt.title('Hierarchical Clustering Dendrogram', fontsize = 28 )
        plt.xlabel('sample index', fontsize = 20)
        plt.ylabel('distance', fontsize = 20)
        #dendrogram(Z)
        output_name = f"Hierarchical_clustering{os.path.splitext(os.path.basename(filename))[0]}.png"
        plt.savefig(os.path.join('results', output_name))
        plt.close()        
        #plt.show()

        # for automatic determination of number of cluster
        n =AutoElbow.auto_elbow_search(df_dum)
        print("optimized number of clusters:",n)

        # prediction of clusters label in given dataframe using Hirarchical clusteing method
        hc = AgglomerativeClustering(n_clusters = n, linkage = 'ward')
        labels = hc.fit_predict(df_dum)
        pred_agc = pd.Series(hc.labels_)
        print(pred_agc)

    #creat dataframe including clusters lable 
        hc_df=pd.DataFrame(df)
        hc_df['hcluster']=pred_agc
        print(hc_df)

    ##visulazing hcluster results
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # Create scatter plot for each cluster with labels
        # for cluster_label in hc_df['hcluster'].unique():
        #     cluster_data = hc_df[hc_df['hcluster'] == cluster_label]
        #     ax.scatter(cluster_data['Age'], cluster_data['Family_Size'], cluster_data['Work_Experience'], label=f'Cluster {cluster_label}')
        #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        # #plt.colorbar(sc)
        # ax.set_title("Age vs Family size vs Work_Experience", fontsize=15)
        # ax.set_xlabel("Age", fontsize=10)
        # ax.set_ylabel("Family_Size", fontsize=10)
        # ax.set_zlabel("Work_Experience", fontsize=10)
        # plt.savefig('results/Age vs family_size vs Work_Experience.png')
        # plt.close()
        #plt.show()
        return hc.labels_
    
    except Exception as e:
        print(f"An error occurred during preprocessing for dataset: {str(e)}")
        return None
    
# def HierarchicalClustering_customer(filename):
#     try:
#         #loading the processed datasets and make dummies columns for categorical feature
#         df=pd.read_csv(filename, index_col=[0])
#         df_dum = pd.get_dummies(df, drop_first=True, dtype=int)
#         #print(df_dum)

#         Z = linkage(df_dum, method='ward', metric='euclidean')

#         # Plot the dendrogram
#         plt.figure(figsize=(25, 25))
#         plt.title('Hierarchical Clustering Dendrogram', fontsize = 14 )
#         plt.xlabel('sample index', fontsize = 10)
#         plt.ylabel('distance', fontsize = 10)
#         dendrogram(Z)
#         output_name = f"Hierarchical_clustering{os.path.splitext(os.path.basename(filename))[0]}.png"
#         plt.savefig(os.path.join('results', output_name))
#         plt.close()
#         #plt.savefig("results/Hierarchical_clustering.png")
#         #plt.show()

#         hierarchical_cluster = AgglomerativeClustering(n_clusters = 4, linkage = 'ward')
#         labels = hierarchical_cluster.fit_predict(df_dum)
#         pred_agc = pd.Series(hierarchical_cluster.labels_)
#         print(pred_agc)

#         # Compute Silhouette Score
#         from sklearn.metrics import silhouette_score
#         score = silhouette_score(df_dum, labels)
#         print("Silhouette Score: ", score)

#     #creat dataframe including cluster lable 
#         hc_df=pd.DataFrame(df)
#         hc_df['hcluster']=pred_agc
#         print(hc_df)

#     ##visulazing hcluster results
#         from mpl_toolkits.mplot3d import Axes3D
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         # Create scatter plot for each cluster with labels
#         for cluster_label in hc_df['hcluster'].unique():
#             cluster_data = hc_df[hc_df['hcluster'] == cluster_label]
#             ax.scatter(cluster_data['Income'], cluster_data['Older'], cluster_data['Recency'], label=f'Cluster {cluster_label}')
#             ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        
#         ax.set_title("Income vs Age vs Recency", fontsize=15)
#         ax.set_xlabel("Income", fontsize=10)
#         ax.set_ylabel("Age", fontsize=10)
#         ax.set_zlabel("Recency", fontsize=10)
#         plt.savefig('results/Income vs Age vs Recency.png')
#         plt.close()
#         #plt.show()
#         return filename
    
#     except Exception as e:
#         print(f"An error occurred during preprocessing for dataset: {str(e)}")
#         return None