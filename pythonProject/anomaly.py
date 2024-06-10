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
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import time
import psutil
# Measure CPU time usage
start_time = time.time()

# Measure memory usage before executing the code snippet
start_memory = psutil.virtual_memory().used

# Measure disk usage before executing the code snippet
start_disk_usage = psutil.disk_usage('/').used

# Define the file path
file_path = r'C:\Dataset.xls'

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)

# Split data into features (X) and target variable (y)
X = df[['V', 'H', 'S']]  # Features: Voltage (V), High (H), Soil Type (S)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Define the autoencoder architecture
input_dim = X_train.shape[1]
encoding_dim = 2  # You can adjust this based on your data
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

# Build the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

# Use the trained autoencoder to reconstruct the instances
reconstructed_instances = autoencoder.predict(X_scaled)

# Calculate reconstruction errors
reconstruction_errors = np.mean(np.square(X_scaled - reconstructed_instances), axis=1)

# Define a threshold for outliers (e.g., based on quantiles or manually)
threshold = np.percentile(reconstruction_errors, 95)  # Adjust as needed

# Identify outliers
outliers = df[reconstruction_errors > threshold]

print("Outliers:")
print(outliers)

# Measure CPU time usage
end_time = time.time()
cpu_time = end_time - start_time
print("CPU Time Usage:", cpu_time)

# Measure memory usage after executing the code snippet
end_memory = psutil.virtual_memory().used
memory_usage = end_memory - start_memory
print("Memory Usage:", memory_usage / (1024 * 1024), "MB")  # Convert bytes to MB

# Measure disk usage after executing the code snippet
end_disk_usage = psutil.disk_usage('/').used
disk_usage = end_disk_usage - start_disk_usage
print("Disk Usage:", disk_usage / (1024 * 1024), "MB")  # Convert bytes to MB