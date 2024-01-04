import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

df = pd.read_csv('DATA/cancer_tumor_data_features.csv')
print(df.head())
print(df.info())
print(df.describe().transpose())

# Scaling Data

scaler = StandardScaler()
scaled_X = scaler.fit_transform(df)
print('scaled_X: ', scaled_X)

# Scikit-Learn Implementation

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_X)

plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0],principal_components[:,1])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

# REQUIRES INTERNET CONNECTION AND FIREWALL ACCESS
cancer_dictionary = load_breast_cancer()
print('cancer_dictionary keys: ', cancer_dictionary.keys())
print('target: ', cancer_dictionary['target'])

plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0],principal_components[:,1],c=cancer_dictionary['target'])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

# Fitted Model Attributes

print('pca.n_components: ', pca.n_components)
print('pca.components_: ', pca.components_)
# In this numpy matrix array, each row represents a principal component, Principal axes in feature space, representing the directions of maximum variance in the data. The components are sorted by explained_variance_.

df_comp = pd.DataFrame(pca.components_,index=['PC1','PC2'],columns=df.columns)
print(df_comp)

plt.figure(figsize=(20,3),dpi=150)
sns.heatmap(df_comp,annot=True)

print('explained_variance_ratio_: ', pca.explained_variance_ratio_)
print('sum: ', np.sum(pca.explained_variance_ratio_))

pca_30 = PCA(n_components=30)
pca_30.fit(scaled_X)

print('explained_variance_ratio_ 30: ', pca_30.explained_variance_ratio_)
print('sum: ', np.sum(pca_30.explained_variance_ratio_))

explained_variance = []
for n in range(1,30):
    pca = PCA(n_components=n)
    pca.fit(scaled_X)
    explained_variance.append(np.sum(pca.explained_variance_ratio_))

plt.figure()
plt.plot(range(1,30),explained_variance)
plt.xlabel("Number of Components")
plt.ylabel("Variance Explained")

plt.show()
