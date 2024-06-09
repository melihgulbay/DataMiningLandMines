import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import log_loss

# Define the file path
file_path = r'C:\Dataset.xls'

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)

# Define the mine types
mine_types = {
    1: 'Null',
    2: 'Anti-Tank',
    3: 'Anti-Personnel',
    4: 'Booby Trapped Anti-personnel',
    5: 'M14 Anti-personnel'
}

# Define the features
features = ['V', 'H', 'S']

# Define colors for each mine type
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Set up the subplot grid
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Loop through each feature
for i, feature in enumerate(features):
    # Prepare the data
    X = df[[feature]]  # Feature
    y = df['M']  # Target variable
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    
    # Assign cluster labels to each data point
    df[f'{feature}_cluster'] = kmeans.labels_
    
    # Scatter plot of the data colored by cluster
    for cluster_num in range(5):
        cluster_points = df[df[f'{feature}_cluster'] == cluster_num]
        axs[i].scatter(cluster_points[feature], cluster_points[f'{feature}_cluster'], c=colors[cluster_num], cmap='viridis', label=f'Cluster {cluster_num+1} ({mine_types[cluster_num+1]})')
    
    axs[i].set_title(f'Clustered by {feature}')
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Cluster')
    axs[i].grid(True)
    axs[i].legend()

plt.tight_layout()
plt.show()