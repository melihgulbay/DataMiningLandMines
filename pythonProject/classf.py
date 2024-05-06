import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# File path
file_path = r'C:\Dataset.xls'

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)
   
# Split the data into features (X) and target variable (y)
X = df[['V', 'H']]  # Features: Voltage (V), High (H)
y = df['M']  # Target variable: M

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict M using the model
y_pred = model.predict(X)

# Plot the actual M values vs. predicted M values
plt.scatter(y, y_pred)
plt.xlabel('Actual M')
plt.ylabel('Predicted M')
plt.title('Actual vs Predicted M values')
plt.show()

# Calculate the coefficients (slope) and intercept of the linear regression line
slope = model.coef_
intercept = model.intercept_
print("Coefficients (Slope):", slope)
print("Intercept:", intercept)
