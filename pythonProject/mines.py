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

# Convert Mine Type codes to actual descriptions
mine_type_map = {1: 'Null', 2: 'Anti-Tank', 3: 'Anti-Personnel',
                 4: 'Booby Trapped Anti-personnel', 5: 'M14 Anti-personnel'}
y = y.map(mine_type_map)

# Plot Voltage vs Height with Mine Type color-coded
plt.figure(figsize=(10, 6))
for mine_type, color in zip(mine_type_map.values(), ['blue', 'green', 'red', 'purple', 'orange']):
    plt.scatter(X[y == mine_type]['V'], X[y == mine_type]['H'], label=mine_type, color=color)

plt.title('Voltage vs Height with Mine Types')
plt.xlabel('Voltage')
plt.ylabel('Height')
plt.legend(title='Mine Type')
plt.grid(True)
plt.show()

