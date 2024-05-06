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
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# fetch dataset
land_mines = fetch_ucirepo(id=763)

  
# Extracting values for histograms
voltages = land_mines.data.features['V']
heights = land_mines.data.features['H']
soil_types = land_mines.data.features['S']
mine_types = land_mines.data.targets

# Combine voltage and height measurements into one feature matrix
X = np.column_stack((voltages, heights))

# Number of clusters
num_clusters = 5

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)  # You can choose any integer value for random_state
kmeans.fit(X)
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

# Define mapping of cluster labels to mine types
label_to_mine = {
    0: 'Null',
    1: 'Anti-Tank',
    2: 'Anti-Personnel',
    3: 'Booby Trapped Anti-personnel',
    4: 'M14 Anti-personnel'
}

# Visualize clusters
plt.figure(figsize=(8, 6))
for label in range(num_clusters):
    cluster_points = X[cluster_labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=label_to_mine[label], alpha=0.5)

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200)
plt.xlabel('Voltage')
plt.ylabel('Height')
plt.title('K-means Clustering')
plt.legend()
plt.show()



