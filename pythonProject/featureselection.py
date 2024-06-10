from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
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

# File path
file_path = r'C:\Dataset.xls'

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)

# Split data into features (X) and target variable (y)
X = df[['V', 'H', 'S']]  # Features: Voltage (V), High (H), Soil Type (S)
y = df['M']  # Target variable: M

k = 3  # Number of features to select
selector = SelectKBest(score_func=f_classif, k=k)

# Fit the selector to the data
selector.fit(X, y)

# Get the indices of the selected features
selected_indices = selector.get_support(indices=True)

# Get the names of the selected features
selected_features = X.columns[selected_indices]

# Print the selected features
print("Selected Features:", selected_features)

# Transform the data to include only the selected features
X_selected = selector.transform(X)

# Convert X_selected to DataFrame
X_selected_df = pd.DataFrame(data=X_selected, columns=selected_features)

# Print the DataFrame with selected features
print(X_selected_df.head())

# Calculate Spearman correlation
spearman_corr = df[['V', 'H', 'S', 'M']].corr(method='spearman')

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Spearman Correlation Matrix')
plt.show()