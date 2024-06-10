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

# Split data into training, validation, and testing sets (60% training, 20% validation, 20% testing)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Initialize and train your ANN model
model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the training set
train_predictions = model.predict(X_train)
# Make predictions on the validation set
val_predictions = model.predict(X_val)
# Make predictions on the testing set
test_predictions = model.predict(X_test)
# Make predictions on all data combined
all_predictions = model.predict(X)

# Generate confusion matrices
train_conf_matrix = confusion_matrix(y_train, train_predictions)
val_conf_matrix = confusion_matrix(y_val, val_predictions)
test_conf_matrix = confusion_matrix(y_test, test_predictions)
all_conf_matrix = confusion_matrix(y, all_predictions)

# Print confusion matrices
print("Training Confusion Matrix:")
print(train_conf_matrix)
print("\nValidation Confusion Matrix:")
print(val_conf_matrix)
print("\nTest Confusion Matrix:")
print(test_conf_matrix)
print("\nAll Data Combined Confusion Matrix:")
print(all_conf_matrix)

# Define a function to plot confusion matrix
def plot_confusion_matrix(matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(train_conf_matrix, "Training Confusion Matrix")
plot_confusion_matrix(val_conf_matrix, "Validation Confusion Matrix")
plot_confusion_matrix(test_conf_matrix, "Test Confusion Matrix")
plot_confusion_matrix(all_conf_matrix, "All Data Combined Confusion Matrix")

# Split data into features (X) and target variable (y)
X = df[['V', 'H']]  # Features: Voltage (V), High (H)
y = df['M']  # Target variable: M

# Split data into training, validation, and testing sets (60% training, 20% validation, 20% testing)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Initialize and train your ANN model
model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the training set
train_predictions = model.predict(X_train)
# Make predictions on the validation set
val_predictions = model.predict(X_val)
# Make predictions on the testing set
test_predictions = model.predict(X_test)
# Make predictions on all data combined
all_predictions = model.predict(X)

# Generate confusion matrices
train_conf_matrix = confusion_matrix(y_train, train_predictions)
val_conf_matrix = confusion_matrix(y_val, val_predictions)
test_conf_matrix = confusion_matrix(y_test, test_predictions)
all_conf_matrix = confusion_matrix(y, all_predictions)

# Print confusion matrices
print("Training Confusion Matrix:")
print(train_conf_matrix)
print("\nValidation Confusion Matrix:")
print(val_conf_matrix)
print("\nTest Confusion Matrix:")
print(test_conf_matrix)
print("\nAll Data Combined Confusion Matrix:")
print(all_conf_matrix)

# Define a function to plot confusion matrix
def plot_confusion_matrix(matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(train_conf_matrix, "Training Confusion Matrix")
plot_confusion_matrix(val_conf_matrix, "Validation Confusion Matrix")
plot_confusion_matrix(test_conf_matrix, "Test Confusion Matrix")
plot_confusion_matrix(all_conf_matrix, "All Data Combined Confusion Matrix")

# Split data into features (X) and target variable (y)
X = df[['V', 'S']]  # Features: Voltage (V), High (H)
y = df['M']  # Target variable: M

# Split data into training, validation, and testing sets (60% training, 20% validation, 20% testing)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Initialize and train your ANN model
model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the training set
train_predictions = model.predict(X_train)
# Make predictions on the validation set
val_predictions = model.predict(X_val)
# Make predictions on the testing set
test_predictions = model.predict(X_test)
# Make predictions on all data combined
all_predictions = model.predict(X)

# Generate confusion matrices
train_conf_matrix = confusion_matrix(y_train, train_predictions)
val_conf_matrix = confusion_matrix(y_val, val_predictions)
test_conf_matrix = confusion_matrix(y_test, test_predictions)
all_conf_matrix = confusion_matrix(y, all_predictions)

# Print confusion matrices
print("Training Confusion Matrix:")
print(train_conf_matrix)
print("\nValidation Confusion Matrix:")
print(val_conf_matrix)
print("\nTest Confusion Matrix:")
print(test_conf_matrix)
print("\nAll Data Combined Confusion Matrix:")
print(all_conf_matrix)

# Define a function to plot confusion matrix
def plot_confusion_matrix(matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(train_conf_matrix, "Training Confusion Matrix")
plot_confusion_matrix(val_conf_matrix, "Validation Confusion Matrix")
plot_confusion_matrix(test_conf_matrix, "Test Confusion Matrix")
plot_confusion_matrix(all_conf_matrix, "All Data Combined Confusion Matrix")

# Split data into features (X) and target variable (y)
X = df[['V']]  # Features: Voltage (V), High (H)
y = df['M']  # Target variable: M

# Split data into training, validation, and testing sets (60% training, 20% validation, 20% testing)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Initialize and train your ANN model
model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the training set
train_predictions = model.predict(X_train)
# Make predictions on the validation set
val_predictions = model.predict(X_val)
# Make predictions on the testing set
test_predictions = model.predict(X_test)
# Make predictions on all data combined
all_predictions = model.predict(X)

# Generate confusion matrices
train_conf_matrix = confusion_matrix(y_train, train_predictions)
val_conf_matrix = confusion_matrix(y_val, val_predictions)
test_conf_matrix = confusion_matrix(y_test, test_predictions)
all_conf_matrix = confusion_matrix(y, all_predictions)

# Print confusion matrices
print("Training Confusion Matrix:")
print(train_conf_matrix)
print("\nValidation Confusion Matrix:")
print(val_conf_matrix)
print("\nTest Confusion Matrix:")
print(test_conf_matrix)
print("\nAll Data Combined Confusion Matrix:")
print(all_conf_matrix)

# Define a function to plot confusion matrix
def plot_confusion_matrix(matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(train_conf_matrix, "Training Confusion Matrix")
plot_confusion_matrix(val_conf_matrix, "Validation Confusion Matrix")
plot_confusion_matrix(test_conf_matrix, "Test Confusion Matrix")
plot_confusion_matrix(all_conf_matrix, "All Data Combined Confusion Matrix")