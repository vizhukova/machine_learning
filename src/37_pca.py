import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

df = pd.read_csv('DATA/cancer_tumor_data_features.csv')

print(df.head())
print(df.info())
print(df.describe())

# Manual Construction of PCA

scaler = StandardScaler()
scaled_X = scaler.fit_transform(df)
print('scaled_X: ', scaled_X)

# Because we scaled the data, this won't produce any change.
# We've left it here because you would need to do this for unscaled data
scaled_X -= scaled_X.mean(axis=0)

# Grab Covariance Matrix
covariance_matrix = np.cov(scaled_X, rowvar=False)

# Get Eigen Vectors and Eigen Values
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# Choose som number of components
num_components=2

# Get index sorting key based on Eigen Values
sorted_key = np.argsort(eigen_values)[::-1][:num_components]

# Get num_components of Eigen Values and Eigen Vectors
eigen_values, eigen_vectors = eigen_values[sorted_key], eigen_vectors[:, sorted_key]

# Dot product of original data and eigen_vectors are the principal component values
# This is the "projection" step of the original points on to the Principal Component
principal_components=np.dot(scaled_X,eigen_vectors)

print(principal_components)

plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0],principal_components[:,1])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

# REQUIRES INTERNET CONNECTION AND FIREWALL ACCESS
cancer_dictionary = load_breast_cancer()

print('keys: ', cancer_dictionary.keys())
print('target: ', cancer_dictionary['target'])

plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0],principal_components[:,1],c=cancer_dictionary['target'])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

plt.show()