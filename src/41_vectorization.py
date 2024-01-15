import numpy as np
import time

a = np.array([1,2,3,4])
print(a)

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print('Vectorized version: ', str(1000*(toc-tic)) + 'ms')

c = 0 
tic = time.time()

# Whenever is possible = avoid explicit for-loops
# it takes MUCH more time for computing 
for i in range(1000000):
    c += a[i]*b[i]

toc = time.time()

print(c)
print('For loop: ', str(1000*(toc-tic)) + 'ms')

A = np.matrix([
    [1,1,1],
    [2,2,2],
    [3,3,3]
])

# sum horizontally
# axis = 0 - vertically
print(A.sum(axis=1))

def basic_sigmoid(x):
    # return 1 / (1 + math.exp(-x))
    return 1 / (1 + np.exp(-x))

# the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds

def image2vector(image):
    v = image.reshape(len(image[0]) * len(image[0][0]) * 3, 1)
    return v

# Implement a function that normalizes each row of the matrix x (to have unit length).
def normalize_rows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x

# Calculates the softmax for each row of the input x.
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)   
    s = x_exp / x_sum   
    return s

# GRADED FUNCTION: L1

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """   
    loss = np.abs(y - yhat).sum()
    return loss

# GRADED FUNCTION: L2

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    result = y - yhat
    loss = np.dot(result, result).sum()  
    return loss