# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 14:24:41 2016

@author: radu
"""

from __future__ import print_function
import cPickle as pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gzip, time


pickle_file = 'notMNIST.pickle'
MNIST_FILE = 'mnist.pkl.gz'

NUM_CLASSES = 10
IMAGE_SIZE = 28  # Pixel width and height.
NUM_CHANNELS = 1 # grayscale

np.random.seed(133)
        

def load_datasets_notMNIST(pickle_file):
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
        
        
def load_datasets_MNIST(pickle_file, verbose=False):
    '''
    Loads the MNIST dataset from a pickle file
    '''
    # Load the dataset
    with gzip.open(pickle_file, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)
        train_dataset, train_labels = train_set[0], train_set[1]
        valid_dataset, valid_labels = valid_set[0], valid_set[1]
        test_dataset, test_labels = test_set[0], test_set[1]
        del train_set, valid_set, test_set  # hint to help gc free up memory
    
    if verbose:
        print ('Training set', train_dataset.shape, train_labels.shape)
        print ('Validation set', valid_dataset.shape, valid_labels.shape)
        print ('Test set', test_dataset.shape, test_labels.shape)
        
    return train_dataset, train_labels, valid_dataset, valid_labels, \
        test_dataset, test_labels
                
        
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(NUM_CLASSES) == labels[:,None]).astype(np.float32)
    return dataset, labels   
                        

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
          
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(count):
    initial = tf.constant(0.1, shape=[count])
    return tf.Variable(initial)
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    
    
    
def tensorflow_conv(train_dataset, train_labels, valid_dataset, valid_labels
                    ,test_dataset, test_labels, save_best_params=False
                    ,batch_size=16, patch_size=5, depth_1=32, depth_2=64
                    ,learn_rate=0.1, reg_param=0.001, keep_probability=0.5
                    ,num_steps=3001, hidden_count=1024):  
    '''
    Tensorflow implementation of classification using Convolutional NN 
    of a single hidden layer and Stochastic Gradient Descent (SGD)
    '''                   
    
    # First we describe the computation that you want to see performed: 
    # what the inputs, the variables, and the operations look like. 
    # These get created as nodes over a computation graph.
    graph = tf.Graph()
    with graph.as_default():
        
        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS])
        tf_train_labels = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)


        # Variables.
        # input layer to conv layer 1
        W_1 = weight_variable([patch_size, patch_size, NUM_CHANNELS, depth_1])
        b_1 = bias_variable(depth_1) 
        # conv layer 1 to conv layer 2
        W_2 = weight_variable([patch_size, patch_size, depth_1, depth_2])
        b_2 = tf.Variable(tf.constant(0.1, shape=[depth_2]))        
        # conv layer 2 to fully connected layer 3 to output layer
        W_3 = weight_variable([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * depth_2, hidden_count])
        b_3 = tf.Variable(tf.constant(0.1, shape=[hidden_count]))
        # dropout layer
        keep_prob = tf.placeholder(tf.float32)                
        # fully connected layer 3 to output layer
        W_4 = weight_variable([hidden_count, NUM_CLASSES])
        b_4 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
        
        
        # Model.
        def model(data):
            # convolutional layer 1
            data = tf.reshape(data, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
            activ = tf.nn.relu(conv2d(data, W_1) + b_1)
            pool = max_pool_2x2(activ)
            # convolutional layer 2
            activ = tf.nn.relu(conv2d(pool, W_2) + b_2)
            pool = max_pool_2x2(activ)
                        
            shape = pool.get_shape().as_list()
            print (shape)
            reshape = tf.reshape(pool, [-1, shape[1] * shape[2] * shape[3]])
            activ = tf.nn.relu(tf.matmul(reshape, W_3) + b_3)
            
            activ = tf.nn.dropout(activ, keep_prob)
            
            return tf.matmul(activ, W_4) + b_4
        
        
        # Training computation.
        logits = model(tf_train_dataset)
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        
        # Optimizer.
        #optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)  
        #optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)
        
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        
        cross_entropy = -tf.reduce_sum(tf_train_labels * tf.log(train_prediction))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(train_prediction,1), tf.argmax(tf_train_labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        
        error_hist = np.array([])
        
        with tf.Session(graph=graph) as session:      
            start = time.time()
            
            # initialize all the variables            
            tf.initialize_all_variables().run()
            
            # train over the configured number of steps
            for step in range(num_steps):
                
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                
                
                #_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                train_accuracy = train_step.run(feed_dict = {
                                                    tf_train_dataset : batch_data, 
                                                    tf_train_labels : batch_labels,
                                                    keep_prob: 0.5
                                                })
                
                if (step % 50 == 0):
                    valid_accuracy = accuracy.eval(feed_dict = {
                                                    tf_train_dataset : valid_dataset, 
                                                    tf_train_labels : valid_labels,
                                                    keep_prob: 1.0
                                                })                    
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.2f%%' % train_accuracy)
                    print('Validation accuracy: %.2f%%\n' % valid_accuracy)
            end = time.time()   
            
            test_accuracy = accuracy.eval(feed_dict = {
                                            tf_train_dataset : test_dataset, 
                                            tf_train_labels : test_labels,
                                            keep_prob: 1.0
                                        }) 
            print('Test accuracy: %.1f%% after %.2f sec\n' 
                % (test_accuracy, (end - start)))
                
            print ('Learning rate: ', learn_rate)
            print ('Regularization rate: ', reg_param)
            print ('Keep probability: ', keep_probability)
            print ('Batch size: ', batch_size)
            print ('Fully connected layer size: ', hidden_count)
                
            plt.plot(error_hist)
            plt.ylabel('Error rates')
            plt.show()
            
            
    
                
if __name__ == '__main__':
    # load the existing data 
    train_dataset, train_labels, \
        valid_dataset, valid_labels, \
        test_dataset, test_labels = load_datasets_MNIST(MNIST_FILE, verbose=True)
        #test_dataset, test_labels = load_datasets_notMNIST(pickle_file)
        
    # reformat the data to a flat matrix and the labels as 1-hot encoding
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    
    tensorflow_conv(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
    print ('Dataset: ', 'MNIST')

