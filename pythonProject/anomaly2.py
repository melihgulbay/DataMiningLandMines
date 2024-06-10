import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import keras
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
from sklearn.ensemble import IsolationForest

# Define the file path
file_path = r'C:\Dataset.xls'

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)

# Split data into features (X) and target variable (y)
X = df[['V', 'H', 'S']]  # Features: Voltage (V), High (H), Soil Type (S)

# Create and fit the Isolation Forest model
isolation_forest = IsolationForest(contamination=0.05)  # Adjust contamination based on expected outlier percentage
isolation_forest.fit(X)

# Predict outliers
outliers = isolation_forest.predict(X)

# Filter out outlier instances
outlier_indices = outliers == -1
outlier_instances = df[outlier_indices]

print("Outliers:")
print(outlier_instances)