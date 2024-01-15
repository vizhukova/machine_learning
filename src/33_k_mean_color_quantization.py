import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image_as_array = mpimg.imread('DATA/palm_trees.jpg')
print(image_as_array) # RGB CODES FOR EACH PIXEL

plt.figure(figsize=(6,6),dpi=200)
plt.imshow(image_as_array)

# Using Kmeans to Quantize Colors
# Quantizing colors means we'll reduce the number of unique colors here to K unique colors. 
print(image_as_array.shape)
# (h,w,3 color channels)

# Convert from 3d to 2d
(h,w,c) = image_as_array.shape
image_as_array2d = image_as_array.reshape(h*w,c)
model = KMeans(n_clusters=6)
print(model)

labels = model.fit_predict(image_as_array2d)
print('labels: ', labels)

# THESE ARE THE 6 RGB COLOR CODES!
print('cluster_centers_: ', model.cluster_centers_)

rgb_codes = model.cluster_centers_.round(0).astype(int)
print('rgb_codes: ', rgb_codes)

quantized_image = np.reshape(rgb_codes[labels], (h, w, c))
print('quantized_image: ', quantized_image)

plt.figure(figsize=(6,6),dpi=200)
plt.imshow(quantized_image)

plt.show()