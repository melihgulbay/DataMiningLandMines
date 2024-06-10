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

# Create subplots for histograms
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot histograms for Voltage (V), Height (H), Soil Type (S), and Mine Type (M)
axs[0, 0].hist(X['V'], bins=10, color='skyblue', edgecolor='black')
axs[0, 0].set_title('Voltage Histogram')
axs[0, 0].set_xlabel('Voltage')
axs[0, 0].set_ylabel('Frequency')

axs[0, 1].hist(X['H'], bins=10, color='lightgreen', edgecolor='black')
axs[0, 1].set_title('Height Histogram')
axs[0, 1].set_xlabel('Height')
axs[0, 1].set_ylabel('Frequency')

# Convert Soil Type codes to actual descriptions
soil_type_map = {0: 'Dry and Sandy', 0.2: 'Dry and Humus', 0.4: 'Dry and Limy',
                 0.6: 'Humid and Sandy', 0.8: 'Humid and Humus', 1: 'Humid and Limy'}
X['S'] = X['S'].map(soil_type_map)

axs[1, 0].hist(X['S'], bins=len(soil_type_map), color='orange', edgecolor='black')
axs[1, 0].set_title('Soil Type Histogram')
axs[1, 0].set_xlabel('Soil Type')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].tick_params(axis='x', rotation=45)

mine_type_map = {1: 'Null', 2: 'Anti-Tank', 3: 'Anti-Personnel',
                 4: 'Booby Trapped Anti-personnel', 5: 'M14 Anti-personnel'}
y = y.map(mine_type_map)

axs[1, 1].hist(y, bins=len(mine_type_map), color='salmon', edgecolor='black')
axs[1, 1].set_title('Mine Type Histogram')
axs[1, 1].set_xlabel('Mine Type')
axs[1, 1].set_ylabel('Frequency')
axs[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()