# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:57:20 2016

@author: Radu
"""

import theano.tensor as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

print (f(2, 3))
print (z.eval({ x : 16.3, y : 12.1 }))
print (type(x))
print (x.type)


x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
print (f([[1, 2], [3, 4]], [[10, 20], [30, 40]]))


x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = function([x], s)
print (logistic([[0, 1], [-1, -2]]))


a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = function([a, b], [diff, abs_diff, diff_squared])
print (f([[1, 1], [1, 1]], [[0, 1], [2, 3]]))


from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
