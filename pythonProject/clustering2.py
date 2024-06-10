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
X = df[['V', 'H', 'S']]  # Features: Voltage (V), High (H), Soil Type (S)
y = df['M']  # Target variable: Mine types (1 to 5)

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Define Mine Type names
mine_type_names = {
    0: 'Null',
    1: 'Anti-Tank',
    2: 'Anti-Personnel',
    3: 'Booby Trapped Anti-personnel',
    4: 'M14 Anti-personnel'
}

# Visualize the clusters in 3D space
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot points colored by cluster label
scatter = ax.scatter(X['V'], X['H'], X['S'], c=labels, cmap='viridis', s=50, alpha=0.5)

# Add annotations for Mine Type names
for i in range(len(kmeans.cluster_centers_)):
    ax.text(kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1], kmeans.cluster_centers_[i][2],
            mine_type_names[i], color='black', fontsize=12, ha='center')

ax.set_xlabel('Voltage')
ax.set_ylabel('High')
ax.set_zlabel('Soil Type')
ax.set_title('K-means Clustering with Mine Type Names')

# Add color bar
plt.colorbar(scatter, ax=ax, label='Cluster')

plt.show()