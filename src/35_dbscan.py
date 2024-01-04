import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

blobs = pd.read_csv('DATA/cluster_blobs.csv')

print(blobs.head())
print(blobs.info())
print(blobs.describe())
sns.scatterplot(data=blobs,x='X1',y='X2')

moons = pd.read_csv('DATA/cluster_moons.csv')
print(moons.head())
print(moons.info())
print(moons.describe())
plt.figure()
sns.scatterplot(data=moons,x='X1',y='X2')

circles = pd.read_csv('DATA/cluster_circles.csv')
print(circles.head())
print(circles.info())
print(circles.describe())
plt.figure()
sns.scatterplot(data=circles,x='X1',y='X2')

# Label Discovery
def display_categories(model,data,title=''):
    labels = model.fit_predict(data)
    plt.figure()
    sns.scatterplot(data=data,x='X1',y='X2',hue=labels,palette='Set1')
    plt.title(title)

# Kmeans Results
model = KMeans(n_clusters = 2)
display_categories(model,moons, 'KMeans 2')

model = KMeans(n_clusters = 3)
display_categories(model,blobs, 'KMeans 3')

model = KMeans(n_clusters = 2)
display_categories(model,circles, 'KMeans 2')

# DBSCAN Results

model = DBSCAN(eps=0.6)
display_categories(model,blobs, 'DBSCAN')

model = DBSCAN(eps=0.15)
display_categories(model,moons, 'DBSCAN')

model = DBSCAN(eps=0.15)
display_categories(model,circles, 'DBSCAN')

plt.show()
