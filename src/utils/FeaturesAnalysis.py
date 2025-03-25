#This code is written by Lokendra Poudel @ Polaron Analytics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 


def FeatureAnalysis(filename):
    try:

        #load the processed datasets
        df=pd.read_csv(filename, index_col=[0])

        #exploring categorical features
        df_cat = df.select_dtypes('object')
        # Calculate number of rows and columns for subplots
        num_cat_columns = len(df_cat.columns)
        num_rows = int(np.ceil(np.sqrt(num_cat_columns)))
        num_cols = int(np.ceil(num_cat_columns / num_rows))
        # Create plot
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 15))
        fig.suptitle('Categorical Features')
        # Flatten axes if necessary
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        # Plot each categorical feature
        for i, (j, ax) in enumerate(zip(df_cat.columns, axes)):
            if i < num_cat_columns:  # Ensure we only plot up to the number of categorical columns
                sns.countplot(data=df_cat, x=j, hue=j, palette='Set2', ax=ax, legend=False)
                ax.set_title(j, fontsize=15, style='italic')
                ax.set_xticks(range(len(df_cat[j].unique())))
                ax.set_xticklabels(labels=df_cat[j].unique(), rotation=90)
                ax.set_ylabel('Count')
                ax.set_xlabel('')
                for container in ax.containers:
                    ax.bar_label(container, label_type='edge')  # Add labels to bars
        # Hide empty subplots and adjust layout
        for ax in axes[num_cat_columns:]:
            ax.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust suptitle position    
        # Saving the plot with a dynamic filename
        output_name = f"CategoricalFeatures_{os.path.splitext(os.path.basename(filename))[0]}.png"
        plt.savefig(os.path.join('results', output_name))
        #plt.show()

        #exploring Numerical features
        df_nume = df.select_dtypes(['float64','int64'])        
        # Calculate number of rows and columns for subplots
        num_nume_columns = len(df_nume.columns)
        num_rows = int(np.ceil(np.sqrt(num_nume_columns)))
        num_cols = int(np.ceil(num_nume_columns / num_rows))
        # Create plot
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 15))
        fig.suptitle('Numerical Features')
        # Flatten axes if necessary
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        # Plot each categorical feature
        for i, (j, ax) in enumerate(zip(df_nume.columns, axes)):
            if i < num_nume_columns:  # Ensure we only plot up to the number of Numerical columns
                sns.histplot(df_nume , x = j, kde = True, ax=ax )
                #sns.countplot(data=df_nume, x=j, hue=j, kde=True, palette='Set2', ax=ax, legend=False)
                ax.set_title(j, fontsize=15, style='italic')
                ax.set_ylabel('Count')
                ax.set_xlabel('')
    
        # Hide empty subplots and adjust layout
        for ax in axes[num_nume_columns:]:
            ax.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust suptitle position
    
        # Saving the plot with a dynamic filename
        output_name = f"NumericalFeatures_{os.path.splitext(os.path.basename(filename))[0]}.png"
        plt.savefig(os.path.join('results', output_name))
        return filename
    except Exception as e:
        print(f"An error occurred during preprocessing for dataset: {str(e)}")
        return None