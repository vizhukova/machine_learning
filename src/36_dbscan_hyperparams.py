import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

two_blobs = pd.read_csv('DATA/cluster_two_blobs.csv')
two_blobs_outliers = pd.read_csv('DATA/cluster_two_blobs_outliers.csv')

sns.scatterplot(data=two_blobs,x='X1',y='X2')

plt.figure()
sns.scatterplot(data=two_blobs_outliers,x='X1',y='X2')

def display_categories(model,data, title=""):
    labels = model.fit_predict(data)
    plt.figure()
    sns.scatterplot(data=data,x='X1',y='X2',hue=labels,palette='Set1')
    plt.title(title)

dbscan = DBSCAN()
display_categories(dbscan,two_blobs, 'DBSCAN')
display_categories(dbscan,two_blobs_outliers, 'DBSCAN')

# Epsilon

# eps : float, default=0.5
#  |      The maximum distance between two samples for one to be considered
#  |      as in the neighborhood of the other. This is not a maximum bound
#  |      on the distances of points within a cluster. This is the most
#  |      important DBSCAN parameter to choose appropriately for your data set
#  |      and distance function.

# Tiny Epsilon --> Tiny Max Distance --> Everything is an outlier (class=-1)
dbscan = DBSCAN(eps=0.001)
display_categories(dbscan,two_blobs_outliers)

# Huge Epsilon --> Huge Max Distance --> Everything is in the same cluster (class=0)
dbscan = DBSCAN(eps=10)
display_categories(dbscan,two_blobs_outliers)

# How to find a good epsilon?
plt.figure(figsize=(10,6),dpi=200)
dbscan = DBSCAN(eps=1)
display_categories(dbscan,two_blobs_outliers)

print('labels_: ', dbscan.labels_)

np.sum(dbscan.labels_ == -1)

print(100 * np.sum(dbscan.labels_ == -1) / len(dbscan.labels_))

# Charting reasonable Epsilon values

outlier_percent = []
number_of_outliers = []

for eps in np.linspace(0.001,10,100):
    
    # Create Model
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(two_blobs_outliers)
    
    # Log Number of Outliers
    number_of_outliers.append(np.sum(dbscan.labels_ == -1))
    
    # Log percentage of points that are outliers
    perc_outliers = 100 * np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)
    
    outlier_percent.append(perc_outliers)
    
plt.figure()
sns.lineplot(x=np.linspace(0.001,10,100),y=outlier_percent)
plt.ylabel("Percentage of Points Classified as Outliers")
plt.xlabel("Epsilon Value")

plt.figure()
sns.lineplot(x=np.linspace(0.001,10,100),y=number_of_outliers)
plt.ylabel("Number of Points Classified as Outliers")
plt.xlabel("Epsilon Value")
plt.xlim(0,1)

# Do we want to think in terms of percentage targeting instead?
plt.figure()
sns.lineplot(x=np.linspace(0.001,10,100),y=outlier_percent)
plt.ylabel("Percentage of Points Classified as Outliers")
plt.xlabel("Epsilon Value")
plt.ylim(0,5)
plt.xlim(0,2)
plt.hlines(y=1,xmin=0,xmax=2,colors='red',ls='--')

# How to find a good epsilon?
dbscan = DBSCAN(eps=0.4)
display_categories(dbscan,two_blobs_outliers)

# Do we want to think in terms of number of outliers targeting instead?

plt.figure()
sns.lineplot(x=np.linspace(0.001,10,100),y=number_of_outliers)
plt.ylabel("Number of Points Classified as Outliers")
plt.xlabel("Epsilon Value")
plt.ylim(0,10)
plt.xlim(0,6)
plt.hlines(y=3,xmin=0,xmax=10,colors='red',ls='--')

# How to find a good epsilon?
dbscan = DBSCAN(eps=0.75)
display_categories(dbscan,two_blobs_outliers)

# Minimum Samples

# |  min_samples : int, default=5
# |      The number of samples (or total weight) in a neighborhood for a point
# |      to be considered as a core point. This includes the point itself.
 
outlier_percent = []

for n in np.arange(1,100):
    
    # Create Model
    dbscan = DBSCAN(min_samples=n)
    dbscan.fit(two_blobs_outliers)
    
    # Log percentage of points that are outliers
    perc_outliers = 100 * np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)
    
    outlier_percent.append(perc_outliers)
    
plt.figure()
sns.lineplot(x=np.arange(1,100),y=outlier_percent)
plt.ylabel("Percentage of Points Classified as Outliers")
plt.xlabel("Minimum Number of Samples")

num_dim = two_blobs_outliers.shape[1]
dbscan = DBSCAN(min_samples=2*num_dim)
display_categories(dbscan,two_blobs_outliers)

num_dim = two_blobs_outliers.shape[1]
dbscan = DBSCAN(eps=0.75,min_samples=2*num_dim)
display_categories(dbscan,two_blobs_outliers)

dbscan = DBSCAN(min_samples=1)
display_categories(dbscan,two_blobs_outliers)

dbscan = DBSCAN(eps=0.75,min_samples=1)
display_categories(dbscan,two_blobs_outliers)

plt.show()