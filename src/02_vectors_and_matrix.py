import numpy as np
from scipy import sparse

mylist = [1, 2, 3]
print(type(mylist))
myarr = np.array(mylist)
print(type(myarr))
my_matrix = [[1,2,3], [4,5,6], [7,8,9]]
print(my_matrix)
print(np.array(my_matrix))

print(list(range(0, 5, 2)))
print(np.arange(0, 5, 2))

print([1, 2] * 2) # === [1, 2, 1, 2]
print(np.array([1, 2]) * 2) # === [2, 4]

# Fulfill with zeros matrix
print(np.zeros((5, 5)))
# fill with 7th the whole array
print(np.full(10, 7))
# fill with 8th the whole matrix
print(np.full((4, 6), 8))
# This will create array of number 5 repeated 8 times
print(np.repeat(5, 8))
# The fastest one is:
a=np.empty(8); a.fill(5)
print(a)
# Fill with 1th the whole array
print(np.ones(5))
# Returns from 0 to 10 evenly spaced numbers
print(np.linspace(0, 10, 3))
# Returns matrix 5x5 with the main diagonal fulfilled with 1
print(np.eye(5))
# Returns a randon array length with value from 0(included) to 1(not included)
print(np.random.rand(1))
print(np.random.rand(5, 6))
# Returns random values including negative
print(np.random.randn(2,3))
# Returns random value from 0 to 100 in matrix 4x5
print(np.randint(0, 101, (4,5)))
# Will always return the same data set, with the repeating run of rand()
np.random.seed(32)
print(np.random.randn(2,3))

# In descending speed order:

# %timeit a=np.empty(10000); a.fill(5)
# 100000 loops, best of 3: 5.85 us per loop

# %timeit a=np.empty(10000); a[:]=5 
# 100000 loops, best of 3: 7.15 us per loop

# %timeit a=np.ones(10000)*5
# 10000 loops, best of 3: 22.9 us per loop

# %timeit a=np.repeat(5,(10000))
# 10000 loops, best of 3: 81.7 us per loop

# %timeit a=np.tile(5,[10000])
# 10000 loops, best of 3: 82.9 us per loop

arr = np.range(0, 11)
print(arr[1:5])
print(arr[5:])
# Changing by broadcasting from 0 to 5 is only available in numpy, with list is not available
arr[0:5] = 100
print(arr)
slice_of_arr = arr[0:5]
print(slice_of_arr)
# Set everything to 99
slice_of_arr[:] = 99
print(slice_of_arr)
# arr values are also changed to 99 from 0 to 5
print(arr)
arr_copy = arr.copy()
# Now all changes would be only insight the arr_copy
arr_copy[:] = 100
console.log(arr_copy)
console.log(arr)

array_2d = [[5, 10, 15], [20, 25, 30], [35, 40, 45]]
# Takes two first rows and columns starting from 1
array_2d[:2, 1:]

arr = np.range(1, 11)
# Returns boolean array
print(arr > 4)
# Filters all values in array and leave only those that > 4
print(arr[arr > 4])
# Applies to every value in array
print(arr + 4)
print(arr - 5)
# Doubles every number in array
print(arr + arr)
# All values are zeros
print(arr - arr)
# Multiply each value on itself
print(arr * arr)
# arr/arr - will return a warning if 0/0
# 0/0 = nan === warning
# 1/0 === division by zero error , in simple python code
# 1/0 === inf + warning, with the numpy

# Create a vector as a row
vector_row = np.array([1, 2, 3])
# Create a vector as a column
vector_column = np.array([[1],
                          [2],
                          [3]])

# there is also a dedicated structure
matrix_object = np.mat([[1, 2],
                        [1, 2],
                        [1, 2]])
# the matrix data structure is not recommended for two reasons.
# First, arrays are the de facto standard data structure of NumPy.
# Second, the vast majority of NumPy operations return arrays, not matrix
# objects.

#
# Sparse Matrix
#
matrix = np.array([[0, 0],
                   [0, 1],
                   [3, 0]])
# Create compressed sparse row (CSR) matrix
matrix_sparse = sparse.csr_matrix(matrix)
# View sparse matrix
print('matrix_sparse: \n', matrix_sparse)

# Create larger matrix
matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# Create compressed sparse row (CSR) matrix
matrix_large_sparse = sparse.csr_matrix(matrix_large)
print('matrix_large_sparse: \n', matrix_large_sparse)

# Create vector
vector = np.array([1, 2, 3, 4, 5, 6])
print('origin vector: ', vector)
# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   ])
print('origin matrix: ', matrix)
print('Select all elements of a vector: ', vector[:])
print('Select everything up to and including the third element: ', vector[:3])
print('Select everything after the third element: ', vector[3:])
print('Select the last element: ', vector[-1])
print('Select the first two rows and all columns of a matrix: ', matrix[:2, :])

#
# Describing a Matrix
#

print('View number of rows and columns: ', matrix.shape)
print('View number of elements (rows * columns): ', matrix.size)
print('View number of dimensions (column size): ', matrix.ndim)

#
# Applying Operations to Elements
#

# Create function that adds 100 to something


def add_100(x): return x + 100


# Create vectorized function
vectorized_add_100 = np.vectorize(add_100)

# Apply function to all elements in matrix
print('100 + matrix = ', vectorized_add_100(matrix))

print('Find max val: ', np.max(matrix))
print('Find min val: ', np.min(matrix))
print('Find maximum element in each column', np.max(matrix, axis=0))
print('Find maximum element in each row', np.max(matrix, axis=1))
print('Find position of max val: ', np.argmin(matrix))
print('Find position of min val: ', np.argmax(matrix))
#
# Calculating the Average, Variance, and Standard Deviation
#
# sum(items) / quantity
print('The average value of matrix: ', np.mean(matrix))
print('The average value of matrix in each column: ', np.mean(matrix, axis=0))
# sum( (item - average)^2) ) / quantity
print('The variance: ', np.var(matrix))
# variance out of the degree 2
print('The  standard deviation: ', np.std(matrix))

# Reshaping the matrix

# print('Reshape of matrix: ', matrix.reshape(2, 6))
# One useful argument in reshape is -1, which effectively means “as many
# as needed,” so reshape(-1, 1) means one row and as many columns as
# needed:
print('reshape with -1: ', matrix.reshape(1, -1))
# Finally, if we provide one integer, reshape will return a 1D array of
# that length:
print('reshape in 1D array with length 12: ', matrix.reshape(9))

print('Transpose a vector or matrix: ', matrix.T)
print('Transpose vector: ', np.array([[1, 2, 3, 4, 5, 6]]).T)

# Transform a matrix into a one row vector / array
print('Flattening of matrix: ', matrix.flatten())

print('Determinant of matrix: ', np.linalg.det(matrix))

print("The main diagonal: ", matrix.diagonal())
print("The diagonal above the main one: ", matrix.diagonal(offset=1))
print("The diagonal below the main one: ", matrix.diagonal(offset=-1))

print('The matrix trace: ', matrix.trace())
print('Return diagonal and sum elements: ', sum(matrix.diagonal()))

eigenvalues, eigenvectors = np.linalg.eig(matrix)
print('Calculating eigenvalues and eigenvectors: ', eigenvalues)
print('Calculating eigenvalues and eigenvectors: ', eigenvectors)

vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])
print('Calculate dot product', np.dot(vector_a, vector_b))
print('Calculate dot product', vector_a @ vector_b)
# if the value is not 0 - it means that 2 vectors are not perpendiculars

# Create matrix
matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])
# Create matrix
matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])
print('Add two matrices: ', np.add(matrix_a, matrix_b))
print('Add two matrices: ', matrix_a + matrix_b)
print('Substract two matrices: ', np.subtract(matrix_a, matrix_b))
print('Substract two matrices: ', matrix_a - matrix_b)
print('Multiply two matrices: ', np.dot(matrix_a, matrix_b))
print('Multiply two matrices: ', matrix_a @ matrix_b)
print('Multiply two matrices element-wise : ', matrix_a * matrix_b)

matr = np.array([[1, 4],
                 [2, 5]])
print('Calculate inverse of matrix: ', np.linalg.inv(matr))
print('Calculate inverse of matrix: ', matr @ np.linalg.inv(matr))

print('Generate random :', np.random.seed(0))
print('Generate random :', np.random.random(3))
print(
    'Generate three random integers between 1 and 10: ',
    np.random.randint(
        0,
        11,
        3))
print(
    'Draw three numbers from a normal distribution with mean 0.0 and standard deviation of 1.0: ',
    np.random.normal(
        0.0,
        1.0,
        3))
print(
    'Draw three numbers from a logistic distribution with mean 0.0 and scale of 1.0: ',
    np.random.logistic(
        0.0,
        1.0,
        3))
print(
    'Draw three numbers greater than or equal to 1.0 and less than 2.0: ',
    np.random.uniform(
        1.0,
        2.0,
        3))
