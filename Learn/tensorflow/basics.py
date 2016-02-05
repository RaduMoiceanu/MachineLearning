# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 16:22:47 2016

@author: radu
"""

import numpy as np
import tensorflow as tf


# Input data.
# -----------
# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)


# Launch the default graph.
with tf.Session() as sess:

    # To run the matmul op we call the session 'run()' method, passing 'product'
    # which represents the output of the matmul op.  This indicates to the call
    # that we want to get the output of the matmul op back.
    #
    # All inputs needed by the op are run automatically by the session.  They
    # typically are run in parallel.
    #
    # The call 'run(product)' thus causes the execution of threes ops in the
    # graph: the two constants and matmul.
    #
    # The output of the op is returned in 'result' as a numpy `ndarray` object.
    result = sess.run(product)
    print result
    # ==> [[ 12.]]
    
    
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print result

# output:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]