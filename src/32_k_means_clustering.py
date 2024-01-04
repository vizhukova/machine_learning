import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("DATA/bank-full.csv")
print(df.head())
print(df.info())
print(df.columns)

plt.figure(figsize=(12,6),dpi=200)
sns.histplot(data=df,x='age')

plt.figure(figsize=(12,6),dpi=200)
sns.histplot(data=df,x='age',hue='loan')

plt.figure(figsize=(12,6),dpi=200)
sns.histplot(data=df,x='pdays')

plt.figure(figsize=(12,6),dpi=200)
sns.histplot(data=df[df['pdays']!=999],x='pdays')

plt.figure(figsize=(12,6),dpi=200)
sns.histplot(data=df,x='duration',hue='contact')
plt.xlim(0,2000)

plt.figure(figsize=(12,6),dpi=200)
sns.countplot(data=df,x='previous',hue='contact')

plt.figure(figsize=(12,6),dpi=200)
sns.countplot(data=df,x='contact')

# df['previous'].value_counts()
print(df['previous'].value_counts().sum()-36954) # 36954 vs. 8257

# Categorical Features
plt.figure(figsize=(12,6),dpi=200)
# https://stackoverflow.com/questions/46623583/seaborn-countplot-order-categories-by-count
sns.countplot(data=df,x='job',order=df['job'].value_counts().index)
plt.xticks(rotation=90)

plt.figure(figsize=(12,6),dpi=200)
# https://stackoverflow.com/questions/46623583/seaborn-countplot-order-categories-by-count
sns.countplot(data=df,x='education',order=df['education'].value_counts().index)
plt.xticks(rotation=90)

plt.figure(figsize=(12,6),dpi=200)
# https://stackoverflow.com/questions/46623583/seaborn-countplot-order-categories-by-count
sns.countplot(data=df,x='education',order=df['education'].value_counts().index,hue='default')
plt.xticks(rotation=90)

plt.figure(figsize=(12,6),dpi=200)
sns.countplot(data=df,x='default')

# THIS TAKES A LONG TIME!
# plt.figure(figsize=(12,6),dpi=200)
# sns.pairplot(df)

# Clustering
X = pd.get_dummies(df)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Creating and Fitting a KMeans Model
# Note of our method choices here:

# fit(X[, y, sample_weight])

# Compute k-means clustering.
# fit_predict(X[, y, sample_weight])

# Compute cluster centers and predict cluster index for each sample.
# fit_transform(X[, y, sample_weight])

# Compute clustering and transform X to cluster-distance space.
# predict(X[, sample_weight])

# Predict the closest cluster each sample in X belongs to.

model = KMeans(n_clusters=2)

cluster_labels = model.fit_predict(scaled_X)
print('cluster_labels: ', cluster_labels)
print('len(scaled_X): ', len(scaled_X))
print('len(cluster_labels): ', len(cluster_labels))
X['Cluster'] = cluster_labels

plt.figure()
sns.heatmap(X.corr())

print(X.corr()['Cluster'])

plt.figure(figsize=(12,6),dpi=200)
X.corr()['Cluster'].iloc[:-1].sort_values().plot(kind='bar')

# Choosing K Value
ssd = []

for k in range(2,10):
    model = KMeans(n_clusters=k)
    model.fit(scaled_X)
    #Sum of squared distances of samples to their closest cluster center.
    ssd.append(model.inertia_)

plt.figure(figsize=(12,6),dpi=200)
plt.plot(range(2,10),ssd,'o--')
plt.xlabel("K Value")
plt.ylabel("Sum of Squared Distances")

print('ssd: ', ssd)

# Change in SSD from previous K value!
print(pd.Series(ssd).diff())

pd.Series(ssd).diff().plot(kind='bar')

plt.show()