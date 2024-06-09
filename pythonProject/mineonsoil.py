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
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Data
data = np.array([
    [0.320241334, 0.181818182, 0, 1],
    [0.28700875, 0.272727273, 0, 1],
    [0.256283622, 0.454545455, 0, 1],
    [0.262839599, 0.545454545, 0, 1],
    [0.240966463, 0.727272727, 0, 1],
    [0.254410486, 0.818181818, 0, 1],
    [0.234924175, 1, 0, 1],
    [0.999998728, 0.090909091, 0.6, 2],
    [0.975829576, 0.272727273, 0.6, 2],
    [0.815708945, 0.363636364, 0.6, 2],
    [0.655588315, 0.545454545, 0.6, 2],
    [0.628398019, 0.636363636, 0.6, 2],
    [0.504531116, 0.818181818, 0.6, 2],
    [0.474319677, 0.909090909, 0.6, 2],
    [0.613292299, 0.181818182, 0.6, 3],
    [0.425981373, 0.363636364, 0.6, 3],
    [0.384621926, 0.454545455, 0.6, 3],
    [0.356495062, 0.636363636, 0.6, 3],
    [0.350685358, 0.727272727, 0.6, 3],
    [0.338368198, 0.909090909, 0.6, 3],
    [0.531721412, 0.181818182, 0.2, 4],
    [0.504531116, 0.272727273, 0.2, 4],
    [0.438065949, 0.454545455, 0.2, 4],
    [0.398791077, 0.545454545, 0.2, 4],
    [0.36253735, 0.727272727, 0.2, 4],
    [0.314199046, 0.818181818, 0.2, 4],
    [0.501509972, 0.181818182, 0.6, 5],
    [0.534742556, 0.272727273, 0.6, 5],
    [0.489425397, 0.454545455, 0.6, 5],
    [0.443805988, 0.545454545, 0.6, 5],
    [0.389727645, 0.727272727, 0.6, 5],
    [0.395044805, 0.818181818, 0.6, 5]
])

# Assigning numerical values to Soil Type and Mine Type
soil_mapping = {0: 0, 0.2: 2, 0.4: 4, 0.6: 6, 0.8: 8, 1: 1}
mine_mapping = {1: 'Null', 2: 'Anti-Tank', 3: 'Anti-Personnel', 4: 'Booby Trapped Anti-personnel', 5: 'M14 Anti-personnel'}

# Mapping Soil Type and Mine Type
soil_types = [soil_mapping[val] for val in data[:, 2]]
mine_types = data[:, 3].astype(int)

# Color mapping for Mine Type
color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange', 5: 'purple'}
colors = [color_map[val] for val in mine_types]

# Calculating correlation coefficient
correlation_coefficient = np.corrcoef(soil_types, mine_types)[0, 1]

# Plotting
plt.figure(figsize=(8, 6))
for mine_type in mine_mapping.keys():
    indices = np.where(mine_types == mine_type)
    plt.scatter(np.array(soil_types)[indices], np.array(mine_types)[indices], c=color_map[mine_type], marker='o', label=f'{mine_mapping[mine_type]} ({mine_type})')

plt.title('Correlation between Soil Type and Mine Type')
plt.xlabel('Soil Type')
plt.ylabel('Mine Type')
plt.xticks(list(soil_mapping.values()), list(soil_mapping.keys()))
plt.yticks(list(mine_mapping.keys()), list(mine_mapping.values()))
plt.legend()
plt.grid(True)
plt.show()

data = np.array([
    [0.320241334, 0.181818182, 0, 1],
    [0.28700875, 0.272727273, 0, 1],
    [0.256283622, 0.454545455, 0, 1],
    [0.262839599, 0.545454545, 0, 1],
    [0.240966463, 0.727272727, 0, 1],
    [0.254410486, 0.818181818, 0, 1],
    [0.234924175, 1, 0, 1],
    [0.999998728, 0.090909091, 0.6, 2],
    [0.975829576, 0.272727273, 0.6, 2],
    [0.815708945, 0.363636364, 0.6, 2],
    [0.655588315, 0.545454545, 0.6, 2],
    [0.628398019, 0.636363636, 0.6, 2],
    [0.504531116, 0.818181818, 0.6, 2],
    [0.474319677, 0.909090909, 0.6, 2],
    [0.613292299, 0.181818182, 0.6, 3],
    [0.425981373, 0.363636364, 0.6, 3],
    [0.384621926, 0.454545455, 0.6, 3],
    [0.356495062, 0.636363636, 0.6, 3],
    [0.350685358, 0.727272727, 0.6, 3],
    [0.338368198, 0.909090909, 0.6, 3],
    [0.531721412, 0.181818182, 0.2, 4],
    [0.504531116, 0.272727273, 0.2, 4],
    [0.438065949, 0.454545455, 0.2, 4],
    [0.398791077, 0.545454545, 0.2, 4],
    [0.36253735, 0.727272727, 0.2, 4],
    [0.314199046, 0.818181818, 0.2, 4],
    [0.501509972, 0.181818182, 0.6, 5],
    [0.534742556, 0.272727273, 0.6, 5],
    [0.489425397, 0.454545455, 0.6, 5],
    [0.443805988, 0.545454545, 0.6, 5],
    [0.389727645, 0.727272727, 0.6, 5],
    [0.395044805, 0.818181818, 0.6, 5]
])   

data = np.array([
    [0.320241334, 0.181818182, 0, 1],
    [0.28700875, 0.272727273, 0, 1],
    [0.256283622, 0.454545455, 0, 1],
    [0.262839599, 0.545454545, 0, 1],
    [0.240966463, 0.727272727, 0, 1],
    [0.254410486, 0.818181818, 0, 1],
    [0.234924175, 1, 0, 1],
    [0.999998728, 0.090909091, 0.6, 2],
    [0.975829576, 0.272727273, 0.6, 2],
    [0.815708945, 0.363636364, 0.6, 2],
    [0.655588315, 0.545454545, 0.6, 2],
    [0.628398019, 0.636363636, 0.6, 2],
    [0.504531116, 0.818181818, 0.6, 2],
    [0.474319677, 0.909090909, 0.6, 2],
    [0.613292299, 0.181818182, 0.6, 3],
    [0.425981373, 0.363636364, 0.6, 3],
    [0.384621926, 0.454545455, 0.6, 3],
    [0.356495062, 0.636363636, 0.6, 3],
    [0.350685358, 0.727272727, 0.6, 3],
    [0.338368198, 0.909090909, 0.6, 3],
    [0.531721412, 0.181818182, 0.2, 4],
    [0.504531116, 0.272727273, 0.2, 4],
    [0.438065949, 0.454545455, 0.2, 4],
    [0.398791077, 0.545454545, 0.2, 4],
    [0.36253735, 0.727272727, 0.2, 4],
    [0.314199046, 0.818181818, 0.2, 4],
    [0.501509972, 0.181818182, 0.6, 5],
    [0.534742556, 0.272727273, 0.6, 5],
    [0.489425397, 0.454545455, 0.6, 5],
    [0.443805988, 0.545454545, 0.6, 5],
    [0.389727645, 0.727272727, 0.6, 5],
        [0.283987607, 0.181818182, 0.2, 1],
    [0.303262478, 0.272727273, 0.2, 1],
    [0.274924175, 0.454545455, 0.2, 1],
    [0.260029894, 0.545454545, 0.2, 1],
    [0.259818455, 0.727272727, 0.2, 1],
    [0.295014984, 0.181818182, 1, 1],
    [0.27199384, 0.272727273, 1, 1],
    [0.271903031, 0.454545455, 1, 1],
    [0.29220528, 0.545454545, 1, 1],
    [0.285347054, 0.727272727, 1, 1],
    [0.30211447, 0.909090909, 1, 1],
    [0.912385552, 0.181818182, 0.8, 2],
    [0.957702712, 0.272727273, 0.8, 2],
    [0.75830721, 0.454545455, 0.8, 2],
    [0.734138058, 0.545454545, 0.8, 2],
    [0.495256245, 0.727272727, 0.8, 2],
    [0.468277389, 0.818181818, 0.8, 2],
    [0.341691591, 0.181818182, 0.4, 1],
    [0.296072182, 0.272727273, 0.4, 1],
    [0.310398761, 0.454545455, 0.4, 1],
    [0.296072182, 0.545454545, 0.4, 1],
    [0.31743163, 0.727272727, 0.4, 1],
    [0.329304766, 0.818181818, 0.4, 1],
    [0.395044805, 0.818181818, 0.6, 5]
])   

# Extracting voltage and soil type columns
voltage = data[:, 0]
soil_type = data[:, 2]

# Defining colors for different soil types
color_map = {
    0: 'orange',    # Dry and Sandy
    0.2: 'green',     # Dry and Humus
    0.4: 'brown',     # Dry and Limy
    0.6: 'blue',      # Humid and Sandy
    0.8: 'purple',    # Humid and Humus
    1: 'red'        # Humid and Limy
}

# Plot
plt.figure(figsize=(8, 6))
for i, soil in enumerate(soil_type):
    plt.scatter(soil, voltage[i], c=color_map[soil], alpha=0.5)

# Custom legend
custom_legend = [plt.Line2D([0], [0], marker='o', color='w', label=f'Soil Type {key}: {value}', 
                             markersize=10, markerfacecolor=color_map[key]) for key, value in color_map.items()]
plt.legend(handles=custom_legend, loc='upper left')

plt.title('Voltage vs. Soil Type')
plt.xlabel('Soil Type')
plt.ylabel('Voltage')
plt.grid(True)
plt.show()