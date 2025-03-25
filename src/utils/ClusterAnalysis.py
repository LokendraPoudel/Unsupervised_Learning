#This code is written by Lokendra Poudel @ Polaron Analytics
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
import os


def ClusteringAnalysis_dataset(filename):
    try:
        #loading the processed datasets and make dummies columns for categorical feature
        df=pd.read_csv(filename, index_col=[0])

        #cluster analysis for different features
        # Iterate over clusters and plot pie charts for gender distribution
        fig = plt.figure(figsize=(8, 5))
        for i in range(3):
            gender_counts = df[df['Clusters'] == i]['Gender'].value_counts()
            ax = fig.add_subplot(1, 3, i + 1)
            ax.pie(x=gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
            ax.set_title('Gender type in Cluster ' + str(i + 1))
        plt.tight_layout()
        plt.savefig("results/gender_vs_clusters.png")

        #Marital status
        fig = plt.figure(figsize = (8,5))
        for i in range(3):
            Ever_Married_counts = df[df['Clusters'] == i]['Ever_Married'].value_counts()
            ax = fig.add_subplot(1,3,i+1)
            ax.pie(x = Ever_Married_counts.values, labels=Ever_Married_counts.index, autopct='%1.1f%%')
            ax.set_title('Marital Status in Cluster' + str(i+1))
        plt.tight_layout()
        plt.savefig("results/maritalstatus_vs_clusters.png")

        #professional
        fig = plt.figure(figsize = (10,10))

        for i in range(0,3):
            ax = fig.add_subplot(1,3,i+1)
            ax = sns.countplot(df[df['Clusters'] == i] , x = 'Profession', hue='Profession', palette="Set1", legend=False)
            ax.set_title('Type of Profession in cluster' + str(i+1) , fontsize = 14)
            ax.set_ylabel('Count')
            ax.set_xlabel(' ')
            ax.set_xticks(range(len(df['Profession'].unique())))
            ax.set_xticklabels(labels=df['Profession'].unique(), rotation=45)
            ax.xaxis.set_major_locator(FixedLocator(range(len(df['Profession'].unique()))))
            for container in ax.containers:
                ax.bar_label(container, label_type='edge', fontsize=8)

        plt.tight_layout()
        plt.savefig("results/professional vs clusters.png")

        #Age
        fig = plt.figure(figsize = (15,5))

        for i in range(0,3):
            ax = fig.add_subplot(1,3,i+1)
            ax = sns.boxplot(df[df['Clusters'] == i] , x = 'Age')
            ax.set_title('Range of Age in Cluster' + str(i+1) , fontsize = 14)
            ax.set_xlabel(' ')
        plt.tight_layout()
        plt.savefig("results/Age vs clusters.png")


        # Graduation
        fig = plt.figure(figsize = (5,5))

        for i in range(3):
            Graduated_counts = df[df['Clusters'] == i]['Graduated'].value_counts()
            ax = fig.add_subplot(1,3,i+1)
            ax.pie(x = Graduated_counts.values, labels=Graduated_counts.index, autopct='%1.1f%%')
            ax.set_title('Graduated in Cluster' + str(i+1))
        plt.tight_layout()
        plt.savefig("results/graduation vs clusters.png")

        # family size
        fig = plt.figure(figsize = (12,8))
        for i in range(3):
            Family_Size_counts = df[df['Clusters'] == i]['Family_Size'].value_counts()
            ax = fig.add_subplot(1,3,i+1)
            ax.pie(x = Family_Size_counts.values, labels=Family_Size_counts.index, autopct='%1.1f%%')
            ax.set_title('Family_Size in Cluster' + str(i+1))
        plt.tight_layout()
        plt.savefig("results/Family_Size vs clusters.png")

        # Spending Score
        fig = plt.figure(figsize = (10,5))

        for i in range(3):
            Spending_Score_counts = df[df['Clusters'] == i]['Spending_Score'].value_counts()
            ax = fig.add_subplot(1,3,i+1)
            ax.pie(x = Spending_Score_counts.values, labels=Spending_Score_counts.index, autopct='%1.1f%%')
            ax.set_title('Marital Status in Cluster' + str(i+1))
        plt.tight_layout()
        plt.savefig("results/Spending_Score vs clusters.png")

        # Work-experience
        fig = plt.figure(figsize = (15,5))

        for i in range(0,3):
            ax = fig.add_subplot(1,3,i+1)
            ax = sns.boxplot(df[df['Clusters'] == i] , x = 'Work_Experience')
            ax.set_title('Working experience in Cluster' + str(i+1) , fontsize = 14)
            ax.set_xlabel(' ')
        plt.tight_layout()
        plt.savefig("results/workexperience vs clusters.png")
        return filename
    except Exception as e:
        print(f"An error occurred during preprocessing for dataset: {str(e)}")
        return None
    return

def ClusteringAnalysis_customer(filename):
    try:
        #loading the clustered datasets 
        df=pd.read_csv(filename, index_col=[0])
        
        #cluster analysis for different features
        # Iterate over clusters and plot pie charts for education distribution
        fig = plt.figure(figsize=(10, 5))
        for i in range(3):
            education_counts = df[df['Clusters'] == i]['Education'].value_counts()
            ax = fig.add_subplot(1, 3, i + 1)
            ax.pie(x=education_counts.values, labels=education_counts.index, autopct='%1.1f%%')
            ax.set_title('Education type in Cluster ' + str(i + 1))
        plt.tight_layout()
        plt.savefig("results/Education_vs_clusters.png")

        #Marital status
        fig = plt.figure(figsize = (10,5))
        for i in range(3):
            marital_status_counts = df[df['Clusters'] == i]['Marital_Status'].value_counts()
            ax = fig.add_subplot(1,3,i+1)
            ax.pie(x = marital_status_counts.values, labels=marital_status_counts.index, autopct='%1.1f%%')
            ax.set_title('Marital Status in Cluster' + str(i+1))
        plt.tight_layout()
        plt.savefig("results/maritalstatus_vs_clusters.png")

        
        #Age
        fig = plt.figure(figsize = (15,6))

        for i in range(0,3):
            ax = fig.add_subplot(1,3,i+1)
            ax = sns.boxplot(df[df['Clusters'] == i] , x = 'Older')
            ax.set_title('Age of people in Cluster' + str(i+1) , fontsize = 14)
            ax.set_xlabel(' ')
        plt.tight_layout()
        plt.savefig("results/Age vs clusters.png")

        #Income
        fig = plt.figure(figsize = (15,6))

        for i in range(0,3):
            ax = fig.add_subplot(1,3,i+1)
            ax = sns.boxplot(df[df['Clusters'] == i] , x = 'Income')
            ax.set_title('Income of people in Cluster' + str(i+1) , fontsize = 14)
            ax.set_xlabel(' ')
        plt.tight_layout()
        plt.savefig("results/Income vs clusters.png")
        
        
        return filename
    
    
    except Exception as e:
        print(f"An error occurred during preprocessing for dataset: {str(e)}")
        return None
    
    return