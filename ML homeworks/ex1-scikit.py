# -*- coding: utf-8 -*-
"""
Machine Learning Online Class - Exercise 1: Linear Regression
using SciKit

@author: Radu
Created on Wed Nov 25 21:36:33 2015
"""
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


DATA_FILE1 = 'data\ex1data1.txt'
DATA_FILE2 = 'data\ex1data2.txt'

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
    
    
# read data file
data = np.loadtxt(DATA_FILE1, delimiter=",")

# turn the vectors into a 1-column array
X = np.array([data[:, 0]]).T
y = np.array([data[:, 1]]).T 

# plot graphic
fig = plot(X, y, 'Population of City in 10,000s', 'Profit in $10,000s')

# apply linear regression
regr = linear_model.LinearRegression()
regr.fit(np.array(X), np.array(y))

print 'Coefficients:', regr.coef_
print 'Intercept:', regr.intercept_
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(X) - y) ** 2))

# plot linear regression function
plt.scatter(X, y, marker='x', s=50,color='red')
plt.plot(X, regr.predict(X),color='blue')
plt.legend(['Linear regression', 'Training data'])
plt.axis((np.min(X)-1, np.max(X)+1, np.min(y)-1, np.max(y)+1))
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = regr.predict([[3.5]]);
print 'For population = 35,000, we predict a profit of ', predict1*10000;
predict2 = regr.predict([[7]]);
print 'For population = 70,000, we predict a profit of ', predict2*10000;