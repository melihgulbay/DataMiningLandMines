import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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

# Define the file path
file_path = r'C:\Dataset.xls'

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)

# Split data into features (X) and target variable (y)
X = df[['V', 'H', 'S']].values  # Features: Voltage (V), High (H), Soil Type (S)
y = df['M'].values  # Target variable: Mine types (1 to 5)

# Normalize the features (optional, but often recommended for K-means)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow method to visualize the optimal number of clusters
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# From the Elbow method, we can see that the optimal number of clusters is around 3 or 4

# Perform K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

# Visualize the clusters on the V-S and H-S graphs
plt.figure(figsize=(12, 6))

# Plot V-S graph
plt.subplot(1, 2, 1)
for i in range(4):  # Assuming 4 clusters
    plt.scatter(X[:, 0][cluster_labels == i], X[:, 2][cluster_labels == i], label=f'Cluster {i+1}')
plt.xlabel('Voltage (V)')
plt.ylabel('Soil Type (S)')
plt.title('Voltage vs Soil Type')
plt.legend()

# Plot H-S graph
plt.subplot(1, 2, 2)
for i in range(4):  # Assuming 4 clusters
    plt.scatter(X[:, 1][cluster_labels == i], X[:, 2][cluster_labels == i], label=f'Cluster {i+1}')
plt.xlabel('Height (H)')
plt.ylabel('Soil Type (S)')
plt.title('Height vs Soil Type')
plt.legend()

plt.tight_layout()
plt.show()






