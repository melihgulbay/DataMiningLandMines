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
# fetch dataset
land_mines = fetch_ucirepo(id=763)

  
# Extracting values for histograms
voltages = land_mines.data.features['V']
heights = land_mines.data.features['H']
soil_types = land_mines.data.features['S']
mine_types = land_mines.data.targets


# Define soil type labels
plt.subplot(2, 2, 2)
soil_type_labels = {
    0: 'Dry and Sandy',
    0.2: 'Dry and Humus',
    0.4: 'Dry and Limy',
    0.6: 'Humid and Sandy',
    0.8: 'Humid and Humus',
    1: 'Humid and Limy'
}

# Plot histogram for soil types
plt.figure(figsize=(8, 6))
plt.hist(soil_types, bins=len(set(soil_types)), color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Soil Types')
plt.xlabel('Soil Type')
plt.ylabel('Frequency')
plt.grid(True)
plt.xticks(list(soil_type_labels.keys()), [soil_type_labels[key] for key in soil_type_labels.keys()], rotation=45)




# Plotting histograms
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(mine_types, bins=range(1, 7), align='left', rwidth=0.8)
plt.title('Histogram of Mine Type')
plt.xlabel('Mine Type')
plt.ylabel('Frequency')
plt.xticks(range(1, 6), ['Null', 'Anti-Tank', 'Anti-Personnel', 'Booby Trapped Anti-personnel', 'M14 Anti-personnel'])


plt.subplot(2, 2, 2)
plt.hist(heights, bins=20)
plt.title('Histogram of Height')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(voltages, bins=20)
plt.title('Histogram of Voltage')
plt.xlabel('Voltage (V)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


