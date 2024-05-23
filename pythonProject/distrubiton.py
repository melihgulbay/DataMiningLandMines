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

df = pd.read_excel(file_path)

# Split data into features (X) and target variable (y)
X = df[['V', 'H', 'S']]  # Features
y = df['M']  # Target variable



# Combine X and y into a single DataFrame
df_combined = pd.concat([X, y], axis=1)

# Group by both 'S' and 'M' and count occurrences
counts = df_combined.groupby(['S', 'M']).size().unstack(fill_value=0)

# Plot the line graph
counts.plot(kind='line', marker='o')
plt.title('Distribution of Mines on Soil Types')
plt.xlabel('Soil Type')
plt.ylabel('Count of Mines')
plt.xticks(counts.index)  # Set x-ticks as the unique soil types
plt.grid(True)
plt.legend(title='Mine Type')
plt.show()