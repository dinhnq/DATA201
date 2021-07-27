import numpy
import pandas
from matplotlib import pyplot

# Vectors and matrices
x = numpy.array([3,4])
y = numpy.array([[3],[4]])

# their size
print(numpy.shape(x))
print(numpy.shape(y))

# numpy doesn't really differentiate between vectors and matrices (and tensors)
#1   2      row: 0, col: 0 => value: 1
#2   3
#3   4      row: 2, col: 0 => value: 3
a = numpy.array([[1,2], [2,3], [3,4]])
val_3_4 = a[2,1]
val__1 = a[:,0]
size__1 = numpy.shape(a[:,0])

# matrix 3x3 with all values = 1
ones = numpy.ones((3,3))
# matrix 3x3 with all values of the diagonal = 1, others = 0
eye = numpy.eye(3)
# matrix 3x4 with all values = 0
zeros = numpy.zeros((3,4))

# matrix 3x3 with diagonal values = 3; 4; 5
diagonal = numpy.diag([3,4,5])

def draw_vec(data, start=None, color=None):
    data = numpy.squeeze(data)
    if start is None:
        start = numpy.array([0,0])
    if color is None:
        color = 'b'

    x = start[0]
    y = start[1]
    dx = data[0]
    dy = data[1]
    pyplot.arrow(x, y, dx=dx, dy=dy, head_width=0.05, head_length=0.1, linewidth=2, color=color, length_includes_head=True)

vector_1_1 = [1,2]
vector_4_3 = [4,3]
vector_5_5 = [5,5]
draw_vec(vector_1_1)
draw_vec(vector_4_3, [1,2], 'r')
draw_vec(vector_5_5, color='g')
pyplot.axis([0,5,0,10])
pyplot.show()

x = numpy.array([1,2])
draw_vec(2*x, color='r')
draw_vec(x)
pyplot.axis([0,5,0,5])
pyplot.show()

x = numpy.array([1,2])
draw_vec(-x, color='r')
draw_vec(x)
pyplot.axis([-5,5,-5,5])
pyplot.show()

# https://machinelearningcoban.com/math/
# vectors have a length + a direction
# Length (using a norm usch as Euclidean length): ||x|| = sqrt(x . x)
# Inner product: x . y = ||x|| * ||y|| * cos(theta)
# Inner product has 2 steps:
#       1. multiple 2 vectors
#       2. give back a scalar
# scalar (dai luong vo huong trong dai so tuyen tinh): dung de dinh nghia 1 khong gian vector
# Co do lon <=> 1 scalar
# Co do lon + co huong <=> 1 vector

# vector: v(x,y)
# x: one scalar
# y: one scalar
# v: one vector

# example:
# 5: one scalar
# tap hop (5,6,7) or mang [5,6,7]: one vector

x_y_4_norm = numpy.array([3,4])
vector_len = numpy.linalg.norm(x_y_4_norm)

# vector multiplication
# 2 vectors
x = numpy.array([1,2])
y = numpy.array([2,2])

# https://www.tutorialspoint.com/numpy/numpy_dot.htm
# returns the dot product of two arrays.
# For 2-D vectors, it is the equivalent to matrix multiplication.
# For 1-D arrays, it is the inner product of the vectors.
product = x*y
inner_product = numpy.dot(x,y)

theta = numpy.arctan2(y[0],y[1])-numpy.arctan2(x[0],x[1])
same_dot_product = numpy.linalg.norm(x) * numpy.linalg.norm(y) * numpy.cos(theta)
print(same_dot_product)

# Linear Regression I
# Given beta, compute points
npoints = 10
beta = numpy.array([2,3])
epsilon = numpy.random.randn(npoints)*0.05
for i in range(npoints):
    tmp = numpy.random.randn()
    x = numpy.array([1, tmp])
    y = numpy.dot(x, beta) + epsilon[i]
    pyplot.plot(x[1], y, 'r.')
pyplot.show()

# Scikit-Learn: machine learning library (ML algorithms in common)
# Tensorflow or Pytorch: deep learning libraries

# 4 steps <=> 4 algorithms:
# 1. decide if you're doing classification/regression
# 2. select types of model
# 3. assemble your data into a matrix
# 4. standardize it
# => 4 steps                                    <=> sklearn.preprocessing.StandardScaler()

# 1 step <=> 1 algorithm
# separate into training, testing, validation   <=> sklearn.model_selection.train_test_split()

# 1 step <=> 1 algorithm
# select a loss function                        <=> sklearn.metrics

# 1 step <=> 1 algorithm
# train the algorithm, using validation set to work out when to stop    <=> fit()

# 1 step <=> 1 algorithm
# test the algorithm with appropriate metrics, decide whether to keep   <=> predict()

from sklearn import linear_model

x = numpy.array([[0,0],[1,1],[2,2]])
y = numpy.array([0,1,2])

# step 1: choose model
reg = linear_model.LinearRegression()
# step 2: fit it = train the algorithm + using validation set to work out when to stop
reg.fit(x, y)

coef = reg.coef_
x_hat = numpy.array([[0.5,0.5]])

# step 3: predit it = test the algorithm + decide whether to keep
y_hat = reg.predict(x_hat)

# preprocessing data
from sklearn import preprocessing
X_train = numpy.array([[1,1],[2,2]])
scaler = preprocessing.StandardScaler().fit(X_train)

# splitting data into testing and training
# or even do cross-validation
from sklearn.model_selection import train_test_split

X = X_train
y = X_train
X_train, X_test, Y_train, Y_test = train_test_split(X, y)

# Some output metrics from sklearn.metrics

# Matrix norm
norm = numpy.linalg.norm(eye)
transpose = a.T

a = numpy.array([[1,2,3],[2,3,4],[3,4,5]])
x = numpy.array([[1,2,5]])
dot_product = numpy.dot(a,x.T)

multiply_transposes = a.T * x.T

# draw 2 vectors
a = numpy.array([[1,2],[2,3]])
x = numpy.array([[2,1]])
y = numpy.dot(a, x.T)

draw_vec(x)
draw_vec(y, color='r')
pyplot.axis([0,5,0,7])
pyplot.show()

a = numpy.eye(2)
x = numpy.array([[2,1]])
y = numpy.dot(a, x.T)

# 2 vectors on the same row
draw_vec(x)
draw_vec(y, color='r')
pyplot.axis([-5,5,-5,5])
pyplot.show()

a = numpy.array([[1,2,3],[2,3,4],[3,4,5]])
b = numpy.random.rand(3,3) # matrix 3x3
product_vector = a*b

a = numpy.random.rand(3,2)
b = numpy.random.rand(2,4)
dot_product = numpy.dot(a,b)
len_arrs = numpy.shape(dot_product)

a = numpy.array([[1,2,3],[2,3,4],[3,4,5]])
a = numpy.array([[1,2,3],[2,3,4],[3,4,5]])
a_inverse = numpy.linalg.pinv(a)
dot_product = numpy.dot(a,a_inverse)

a = numpy.random.rand(3,2)
#a_inverse = numpy.linalg.inv(a)
#a_pseudo_inv = numpy.linalg.pinv(a)

a = numpy.eye(3)
a_inverse = numpy.linalg.inv(a)

diagonal = numpy.diag([3,4,5])
diagonal_inv = numpy.linalg.inv(diagonal)

# Scatter plots by equation
beta = numpy.array([[2],[3]])
x = numpy.random.rand(30,1)     # matrix 30x1
y = beta[0]*x + beta[1]         # y = 2x + 3
z = y + 0.3*numpy.random.rand(30,1)
pyplot.plot(x,z,'b.')
pyplot.figure()
pyplot.show()

pyplot.plot(x,y,'r.')
pyplot.show()

# Scatter plot
npoints = 20
beta = numpy.array([[2,3]])
x = numpy.random.rand(2,npoints)
x[0,:] = 1
y = numpy.dot(beta,x) + 0.5*numpy.random.rand(1, npoints)

# Least square
from scipy.linalg import lstsq

betahat, r, rank, sigma = lstsq(x.T,y.T)
pyplot.plot(x[1,:], y.T, 'b.')  # y.T = y lat nguoc lai de ra mat phang

newx = numpy.linspace(0,1,num=50)   #one dimension space
newX = numpy.ones((2,50))
newX[1,:] = newx
pyplot.plot(newx, numpy.dot(betahat.T,newX)[0,:])
pyplot.show()

print(betahat)

x_pseudo_inverse = numpy.linalg.pinv(x)
betahat = numpy.dot(x_pseudo_inverse.T,y.T)
print(betahat)