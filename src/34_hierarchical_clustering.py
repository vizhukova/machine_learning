import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy

df = pd.read_csv('DATA/cluster_mpg.csv')

print(df.head())
print(df.info())
print(df.describe())

print(df.isna().sum())
# df = df.dropna()

print(df['origin'].value_counts())

df_w_dummies = pd.get_dummies(df.drop('name',axis=1))
print(df_w_dummies)

# (x1_a - x2_b)^2
# max value can be 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_w_dummies)
print(scaled_data)

scaled_df = pd.DataFrame(scaled_data,columns=df_w_dummies.columns)

plt.figure(figsize=(15,8))
sns.heatmap(scaled_df,cmap='magma')

plt.figure(figsize=(15,8))
sns.clustermap(scaled_df,row_cluster=False)

plt.figure(figsize=(15,8))
sns.clustermap(scaled_df,col_cluster=False)

model = AgglomerativeClustering(n_clusters=4)
cluster_labels = model.fit_predict(scaled_df)
print(cluster_labels)

plt.figure(figsize=(12,4),dpi=200)
sns.scatterplot(data=df,x='mpg',y='weight',hue=cluster_labels)

# Exploring Number of Clusters with Dendrograms

model = AgglomerativeClustering(n_clusters=None,distance_threshold=0)
cluster_labels = model.fit_predict(scaled_df)
print(cluster_labels)

# Linkage Model

linkage_matrix = hierarchy.linkage(model.children_)
print('linkage_matrix: ', linkage_matrix)

plt.figure(figsize=(20,10))
# Warning! This plot will take awhile!!
dn = hierarchy.dendrogram(linkage_matrix)

plt.figure(figsize=(20,10))
dn = hierarchy.dendrogram(linkage_matrix,truncate_mode='lastp',p=48)

# Choosing a Threshold Distance
# What is the distance between two points?

print('scaled_df.describe(): ', scaled_df.describe())
print('idxmax: ', scaled_df['mpg'].idxmax())
print('idxmin: ',scaled_df['mpg'].idxmin())

a = scaled_df.iloc[320]
b = scaled_df.iloc[28]
dist = np.linalg.norm(a-b)

print('dist: ', dist)

# Max possible distance?

print('max dist: ', np.sqrt(len(scaled_df.columns)))

# Creating a Model Based on Distance Threshold
# if the distance between two points equal distance_threshold we still not gonna merge them into one cluster
model = AgglomerativeClustering(n_clusters=None,distance_threshold=2)
cluster_labels = model.fit_predict(scaled_data)
print('cluster_labels: ', cluster_labels)
print('unique: ', np.unique(cluster_labels))

# Linkage Matrix

# A (n-1) by 4 matrix Z is returned. At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster n + i. A cluster with an index less than n corresponds to one of the original observations. The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.

linkage_matrix = hierarchy.linkage(model.children_)
# show two clusters that lined together, distance metrics between them and number of points that connected under that clusters
print('linkage_matrix: ', linkage_matrix)

plt.figure(figsize=(20,10))
dn = hierarchy.dendrogram(linkage_matrix,truncate_mode='lastp',p=11)

plt.show()


