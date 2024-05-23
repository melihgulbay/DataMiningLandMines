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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# File path
file_path = r'C:\Dataset.xls'

df = pd.read_excel(file_path)

# Split data into features (X) and target variable (y)
X = df[['V', 'H']]  # Features
y = df['M']  # Target variable

# Visualizing linear regression between V and H with M
sns.lmplot(x='V', y='H', data=df, hue='M', fit_reg=True)
plt.title('Linear Regression between V and H with M')
plt.xlabel('V')
plt.ylabel('H')
plt.show()