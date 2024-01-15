import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import zipfile
import tarfile
import os

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# 
# Load and Process the Dataset
# 

# Download the test dataset first 
# https://github.com/vizhukova/machine_learning?tab=readme-ov-file#download-the-test-datasets
cats_and_dogs_zip = './archives/kagglecatsanddogs_3367a.zip'
caltech_birds_tar = './archives/CUB_200_2011.tar'

base_dir = './tmp/data'

if not os.path.isdir(base_dir): 
    with zipfile.ZipFile(cats_and_dogs_zip, 'r') as my_zip:
        my_zip.extractall(base_dir)

    with tarfile.open(caltech_birds_tar, 'r') as my_tar:
        my_tar.extractall(base_dir)

base_dogs_dir = os.path.join(base_dir, 'PetImages/Dog')
base_cats_dir = os.path.join(base_dir,'PetImages/Cat')

print(f"There are {len(os.listdir(base_dogs_dir))} images of dogs")
print(f"There are {len(os.listdir(base_cats_dir))} images of cats")

train_x_orig, train_y, test_x_orig, test_y, classes = base_cats_dir

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# print ("Number of training examples: " + str(m_train))
# print ("Number of testing examples: " + str(m_test))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_x_orig shape: " + str(train_x_orig.shape))
# print ("train_y shape: " + str(train_y.shape))
# print ("test_x_orig shape: " + str(test_x_orig.shape))
# print ("test_y shape: " + str(test_y.shape))

# # Reshape the training and test examples 
# train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
# test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# # Standardize data to have feature values between 0 and 1.
# train_x = train_x_flatten/255.
# test_x = test_x_flatten/255.

# # Model Architecture
# # Two-layer Neural Network

# def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
#     """
#     Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
#     Arguments:
#     X -- input data, of shape (n_x, number of examples)
#     Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
#     layers_dims -- dimensions of the layers (n_x, n_h, n_y)
#     num_iterations -- number of iterations of the optimization loop
#     learning_rate -- learning rate of the gradient descent update rule
#     print_cost -- If set to True, this will print the cost every 100 iterations 
    
#     Returns:
#     parameters -- a dictionary containing W1, W2, b1, and b2
#     """
    
#     np.random.seed(1)
#     grads = {}
#     costs = [] # to keep track of the cost
#     m = X.shape[1] # number of examples
#     (n_x, n_h, n_y) = layers_dims
    
#     # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
#     parameters = initialize_parameters(n_x, n_h, n_y)
    
#     # Get W1, b1, W2 and b2 from the dictionary parameters.
#     W1 = parameters["W1"]
#     b1 = parameters["b1"]
#     W2 = parameters["W2"]
#     b2 = parameters["b2"]
    
#     # Loop (gradient descent)

#     for i in range(0, num_iterations):
#         # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
#         A1, cache1 = linear_activation_forward(X, W1, b1, activation = 'relu')
#         A2, cache2 = linear_activation_forward(A1, W2, b2, activation = 'sigmoid')
        
#         cost = compute_cost(A2, Y)
        
#         # Initializing backward propagation
#         dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
#         # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
#         dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation = 'sigmoid')
#         dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation = 'relu')
        
#         # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
#         grads['dW1'] = dW1
#         grads['db1'] = db1
#         grads['dW2'] = dW2
#         grads['db2'] = db2
        
#         # Update parameters.
#         parameters = update_parameters(parameters, grads, learning_rate)
        
#         # Retrieve W1, b1, W2, b2 from parameters
#         W1 = parameters["W1"]
#         b1 = parameters["b1"]
#         W2 = parameters["W2"]
#         b2 = parameters["b2"]
        
#         # Print the cost every 100 iterations
#         if print_cost and i % 100 == 0 or i == num_iterations - 1:
#             print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
#         if i % 100 == 0 or i == num_iterations:
#             costs.append(cost)

#     return parameters, costs

# def plot_costs(costs, learning_rate=0.0075):
#     plt.plot(np.squeeze(costs))
#     plt.ylabel('cost')
#     plt.xlabel('iterations (per hundreds)')
#     plt.title("Learning rate =" + str(learning_rate))
#     plt.show()


# # 
# # Train the model
# # 

# parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
# plot_costs(costs, learning_rate)

# predictions_train = predict(train_x, train_y, parameters)
# predictions_test = predict(test_x, test_y, parameters)

# # 
# # L-layer Neural Network
# # 

# layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

# def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
#     """
#     Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
#     Arguments:
#     X -- input data, of shape (n_x, number of examples)
#     Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
#     layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
#     learning_rate -- learning rate of the gradient descent update rule
#     num_iterations -- number of iterations of the optimization loop
#     print_cost -- if True, it prints the cost every 100 steps
    
#     Returns:
#     parameters -- parameters learnt by the model. They can then be used to predict.
#     """

#     np.random.seed(1)
#     costs = [] # keep track of cost
    
#     # Parameters initialization.
#     parameters = initialize_parameters_deep(layers_dims)
    
#     # Loop (gradient descent)
#     for i in range(0, num_iterations):

#         # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
#         AL, caches = L_model_forward(X, parameters)
        
#         # Compute cost.
#         cost = compute_cost(AL, Y)
        
#         # Backward propagation.
#         grads =  L_model_backward(AL, Y, caches)
        
#         # Update parameters.
#         parameters = update_parameters(parameters, grads, learning_rate)
        
#         # Print the cost every 100 iterations
#         if print_cost and i % 100 == 0 or i == num_iterations - 1:
#             print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
#         if i % 100 == 0 or i == num_iterations:
#             costs.append(cost)
    
#     return parameters, costs

# # Train the model

# parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
# pred_train = predict(train_x, train_y, parameters)
# pred_test = predict(test_x, test_y, parameters)

# # Results Analysis

# print_mislabeled_images(classes, test_x, test_y, pred_test)

# # Test with your own image (optional/ungraded exercise)

# ## START CODE HERE ##
# my_image = "my_image.jpg" # change this to the name of your image file 
# my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
# ## END CODE HERE ##

# fname = "images/" + my_image
# image = np.array(Image.open(fname).resize((num_px, num_px)))
# plt.imshow(image)
# image = image / 255.
# image = image.reshape((1, num_px * num_px * 3)).T

# my_predicted_image = predict(image, my_label_y, parameters)


# print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
