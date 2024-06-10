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

# Combine features and target into a single DataFrame for plotting
data = pd.concat([X, y], axis=1)

# Create boxplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# List of features
features = ['V', 'H', 'S']

for i, feature in enumerate(features):
    data.boxplot(column=feature, by='M', ax=axes[i])
    axes[i].set_title(f'Boxplot of {feature} by Mine Type')
    axes[i].set_xlabel('Mine Type')
    axes[i].set_ylabel(feature)

plt.suptitle('')  # Suppress the default title to avoid overlap
plt.tight_layout()
plt.show()