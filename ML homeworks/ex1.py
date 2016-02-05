# -*- coding: utf-8 -*-
"""
Machine Learning Online Class - Exercise 1: Linear Regression

@author: Radu
Created on Wed Nov 25 21:36:33 2015
"""
import numpy as np
import matplotlib.pyplot as plt


DATA_FILE1 = 'data\ex1data1.txt'
DATA_FILE2 = 'data\ex1data2.txt'


def warmupExercise():
    d = np.eye(5)
    print d


def plot(x, y, x_label=None, y_label=None, title=None):
    '''
    Plots a simple scatter plot graphic
    '''
    fig = plt.scatter(x, y, marker='x', s=50)
    plt.axis((np.min(x)-1, np.max(x)+1, np.min(y)-1, np.max(y)+1))
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    plt.show()
    return fig
    
    
def plot_surface(X, y):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    theta_0_vals = np.arange(-1, 4, 0.1)
    theta_1_vals = np.arange(-10, 10, 0.1)
    
    Z = np.zeros((len(theta_0_vals), len(theta_1_vals)))
    for i in range(0, len(theta_0_vals)):    
        for j in range(0, len(theta_1_vals)):
            Z[i][j] = computeCost(X, y, [[theta_0_vals[i]], [theta_1_vals[j]]])
            
    theta_0_vals, theta_1_vals = np.meshgrid(theta_1_vals, theta_0_vals)
    
    fig2 = plt.figure()
    ax = fig2.gca(projection='3d')
    surf = ax.plot_surface(theta_0_vals, theta_1_vals, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig2.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    plt.figure()
    plt.contourf(theta_0_vals, theta_1_vals, Z)
    plt.show()
   
   
def hypothesis(X, theta):
    '''
    Calculates the hypothesis function for linear regression
    '''
    return np.dot(X, theta)
    

def computeCost(X, y, theta):
    '''
    Computes the cost function for linear regression
    '''    
    m = len(y)
    # square errors for all m examples
    square_errors = (hypothesis(X, theta) - y)**2
    # overall cost function
    J = (1. / (2 * m)) * np.sum(square_errors)
    return J
    
    
def gradientDescent(X, y, theta, alpha, iterations, reg_param):
    '''
    Performs gradient descent to calculate theta
    '''
    m = len(y)
    
    # update theta terms over number of iterations
    for i in range(iterations):        
        temp_0 = theta[0] - alpha * (1. / m) * np.sum((hypothesis(X, theta) - y) * X[:,0])
        temp_1 = theta[1] - alpha * (1. / m) * np.sum((hypothesis(X, theta) - y) * X[:,1]) + ((reg_param / (2 * m)) * theta[1] ** 2)
        # update theta at the same time (otherwise theta[0] will influence theta[1])
        theta[0] = temp_0
        theta[1] = temp_1
    
    return theta


# read data file
data = np.loadtxt(DATA_FILE1, delimiter=",")

# plot graphic
X = data[:, 0]
y = data[:, 1]
fig = plot(X, y, 'Population of City in 10,000s', 'Profit in $10,000s')

# train linear regression
m = len(y)
X = np.array([np.ones(m), data[:, 0]]).T
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500;
alpha = 0.01;
reg_param = 0; # 0.9;

# display initial cost
computeCost(X, y, theta)

# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations, reg_param);
print 'Theta found by gradient descent: ', theta
print("Residual sum of squares: %.2f"
      % np.mean((hypothesis(X, theta) - y) ** 2))

# plot linear regression function
plt.scatter(X[:,1], y, marker='x', s=50,color='red')
plt.plot(X[:,1], hypothesis(X, theta),color='blue')
plt.legend(['Linear regression', 'Training data'])
plt.axis((np.min(X[:,1])-1, np.max(X[:,1])+1, np.min(y)-1, np.max(y)+1))
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = hypothesis([1, 3.5], theta);
print 'For population = 35,000, we predict a profit of ', predict1*10000;
predict2 = hypothesis([1, 7], theta);
print 'For population = 70,000, we predict a profit of ', predict2*10000;

# plot 3D surface of cost function
plot_surface(X, y)