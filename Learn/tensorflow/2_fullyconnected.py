# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 15:48:21 2016

@author: radu
"""

import math
import cPickle as pickle
import numpy as np
import tensorflow as tf
import time


pickle_file = 'notMNIST.pickle'

NUM_CLASSES = 10
IMAGE_SIZE = 28  # Pixel width and height.

np.random.seed(133)



def load_datasets(pickle_file):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        #print 'Training set', train_dataset.shape, train_labels.shape
        #print 'Validation set', valid_dataset.shape, valid_labels.shape
        #print 'Test set', test_dataset.shape, test_labels.shape
    return train_dataset, train_labels, valid_dataset, valid_labels, \
        test_dataset, test_labels
        
        
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(NUM_CLASSES) == labels[:,None]).astype(np.float32)
    return dataset, labels        
        


def tensorflow_gd(train_dataset, train_labels, valid_dataset, valid_labels
                    ,test_dataset, test_labels
                    ,learn_rate=0.5, num_steps = 801):  
    '''
    Tensorflow implementation of logistic regression using regular Gradient Descent
    '''                        
                        
    # With gradient descent training, even this much data is prohibitive.
    # Subset the training data for faster turnaround.
    train_subset = 10000
    
    # First we describe the computation that you want to see performed: 
    # what the inputs, the variables, and the operations look like. 
    # These get created as nodes over a computation graph.
    graph = tf.Graph()
    with graph.as_default():
       
        # Input data.
        # -----------
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)


        # Variables.
        # ----------
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random valued following a (truncated)
        # normal distribution. The biases get initialized to zero.
        weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, NUM_CLASSES]))
        biases = tf.Variable(tf.zeros([NUM_CLASSES]))   
        
        
        # Training computation.
        # ---------------------
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        
        
        # Optimizer.
        # ----------
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
        
        
        # Predictions for the training, validation, and test data.
        # --------------------------------------------------------
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
        
        
        # you can run the operations on this graph as many times as you want by 
        # calling session.run(), providing it outputs to fetch from the graph that get returned.
        with tf.Session(graph=graph) as session:
            start = time.time()
        
            # This is a one-time operation which ensures the parameters get initialized as
            # we described in the graph: random weights for the matrix, zeros for the
            # biases. 
            tf.initialize_all_variables().run()
            print 'Initialized'
            
            for step in xrange(num_steps):
                
                # Run the computations. 
                # ---------------------
                # We tell .run() that we want to run the optimizer and get the loss value 
                # and the training predictions returned as numpy arrays.
                _, l, predictions = session.run([optimizer, loss, train_prediction])
                if (step % 100 == 0):
                    print 'Loss at step', step, ':', l
                    print 'Training accuracy: %.1f%% ' % \
                        accuracy(predictions, train_labels[:train_subset, :])
                
                    # Calling .eval() on valid_prediction is basically like calling run(), but
                    # just to get that one numpy array. Note that it recomputes all its graph
                    # dependencies.
                    print 'Validation accuracy: %.1f%%\n' % \
                            accuracy(valid_prediction.eval(), valid_labels)
                   
            end = time.time()         
            print 'Test accuracy: %.1f%% after %.2f sec \n\n' % \
                    (accuracy(test_prediction.eval(), test_labels), (end - start))
                      


def tensorflow_sgd(train_dataset, train_labels, valid_dataset, valid_labels
                    ,test_dataset, test_labels
                    ,learn_rate=0.5, num_steps = 3001, batch_size = 128):  
    '''
    Tensorflow implementation of logistic regression using Stochastic Gradient Descent (SGD)
    '''                         
    
    # First we describe the computation that you want to see performed: 
    # what the inputs, the variables, and the operations look like. 
    # These get created as nodes over a computation graph.
    graph = tf.Graph()
    with graph.as_default():
        
        # Input data. 
        # -----------
        # For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE * IMAGE_SIZE))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, NUM_CLASSES))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)


        # Variables.
        # ----------
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random valued following a (truncated)
        # normal distribution. The biases get initialized to zero.
        weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, NUM_CLASSES]))
        biases = tf.Variable(tf.zeros([NUM_CLASSES]))   
        
        
        # Training computation.
        # ---------------------
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        
        
        # Optimizer.
        # ----------
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
        
        
        # Predictions for the training, validation, and test data.
        # --------------------------------------------------------
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
        
        
        # you can run the operations on this graph as many times as you want by 
        # calling session.run(), providing it outputs to fetch from the graph that get returned.
        with tf.Session(graph=graph) as session:
            start = time.time()
        
            # This is a one-time operation which ensures the parameters get initialized as
            # we described in the graph: random weights for the matrix, zeros for the
            # biases. 
            tf.initialize_all_variables().run()
            print 'Initialized'
            
            for step in xrange(num_steps):
                
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                
                # Generate a minibatch.
                batch_data = train_dataset[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
    
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {
                    tf_train_dataset : batch_data, 
                    tf_train_labels : batch_labels
                    }
                    
                    
                # Run the computations. 
                # ---------------------
                # We tell .run() that we want to run the optimizer and get the loss value 
                # and the training predictions returned as numpy arrays.
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                
                if (step % 500 == 0):
                    print "Minibatch loss at step", step, ":", l
                    print "Minibatch accuracy: %.1f%%" \
                        % accuracy(predictions, batch_labels)
                    print "Validation accuracy: %.1f%% \n" % \
                        accuracy(valid_prediction.eval(), valid_labels)
                
            end = time.time()     
            print "Test accuracy: %.1f%% after %.2f sec\n" % \
                (accuracy(test_prediction.eval(), test_labels), (end - start))
                
            print 'Learning rate: ', learn_rate   
               
        

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])
        
        
        
if __name__ == '__main__':    
    # load the existing data 
    train_dataset, train_labels, \
        valid_dataset, valid_labels, \
        test_dataset, test_labels = load_datasets(pickle_file)
        
    # reformat the data to a flat matrix and the labels as 1-hot encoding
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    #print 'Training set', train_dataset.shape, train_labels.shape
    #print 'Validation set', valid_dataset.shape, valid_labels.shape
    #print 'Test set', test_dataset.shape, test_labels.shape
           
    #tensorflow_gd(train_dataset, train_labels, valid_dataset, valid_labels, 
    #               test_dataset, test_labels) 
    tensorflow_sgd(train_dataset, train_labels, valid_dataset, valid_labels, 
                   test_dataset, test_labels, num_steps=10001,
                   learn_rate=1.0) 
     
    