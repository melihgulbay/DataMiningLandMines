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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Given data
voltage = [0.363636364, 0.545454545, 0.636363636, 0.818181818, 0.909090909, 0,
          0.090909091, 0.272727273, 0.363636364, 0.545454545]
height = [0.815708945, 0.655588315, 0.628398019, 0.504531116, 0.474319677, 0.719032338,
           0.622355731, 0.531721412, 0.504531116, 0.438065949]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(height, voltage, marker='o', linestyle='-', color='b')
plt.title('Anti-Tank Mine Type Distribution on Voltage-Height Graph')
plt.xlabel('Height')
plt.ylabel('Voltage')
plt.grid(True)
plt.show()

# Given data
voltage_null = [0, 0.181818182, 0.272727273, 0.454545455, 0.545454545, 0.727272727,
                0.818181818, 1, 0]
height_null = [0.338156758, 0.320241334, 0.28700875, 0.256283622, 0.262839599,
               0.240966463, 0.254410486, 0.234924175, 0]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(height_null, voltage_null, marker='o', linestyle='-', color='r')
plt.title('Null Type Mine Distribution on Voltage-Height Graph')
plt.xlabel('Height')
plt.ylabel('Voltage')
plt.grid(True)
plt.show()

# Given data
height_anti_personnel = [0.181818182, 0.272727273, 0.454545455, 0.545454545, 0.727272727,
                         0.818181818, 0.181818182, 0.363636364, 0.454545455, 0.636363636,
                         0.727272727, 0.909090909]
voltage_anti_personnel = [0.365347054, 0.368579638, 0.338156758, 0.31722019, 0.319093326,
                          0.335347054, 0.613292299, 0.425981373, 0.384621926, 0.356495062,
                          0.350685358, 0.338368198]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(height_anti_personnel, voltage_anti_personnel, marker='o', linestyle='-', color='g')
plt.title('Anti-Personnel Mine Type Distribution on Voltage-Height Graph')
plt.xlabel('Height')
plt.ylabel('Voltage')
plt.grid(True)
plt.show()


# Given data
height_booby_trapped = [0.181818182, 0.272727273, 0.454545455, 0.545454545, 0.727272727,
                        0.818181818, 0.090909091, 0.181818182, 0.454545455, 0.545454545,
                        0.727272727, 0.818181818]
voltage_booby_trapped = [0.273866976, 0.256465241, 0.227522439, 0.235649303, 0.207945319,
                         0.197733879, 0.24220528, 0.240332143, 0.262839599, 0.255860743,
                         0.233987607, 0.213096646]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(height_booby_trapped, voltage_booby_trapped, marker='o', linestyle='-', color='m')
plt.title('Booby-Trapped Mine Type Distribution on Voltage-Height Graph')
plt.xlabel('Height')
plt.ylabel('Voltage')
plt.grid(True)
plt.show()

# Given data
height_m14_anti_personnel = [0.090909091, 0.181818182, 0.363636364, 0.454545455, 0.636363636,
                              0.727272727, 0.909090909, 0.181818182, 0.272727273, 0.454545455,
                              0.545454545, 0.727272727, 0.818181818]
voltage_m14_anti_personnel = [0.419184136, 0.422839599, 0.378368198, 0.371600782, 0.348791077,
                               0.353685358, 0.323262478, 0.471298533, 0.444108237, 0.383685358,
                               0.350875653, 0.341389342, 0.341389342]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(height_m14_anti_personnel, voltage_m14_anti_personnel, marker='o', linestyle='-', color='c')
plt.title('M14 Anti-Personnel Mine Type Distribution on Voltage-Height Graph')
plt.xlabel('Height')
plt.ylabel('Voltage')
plt.grid(True)
plt.show()

