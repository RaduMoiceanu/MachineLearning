# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 14:24:41 2016

@author: radu
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from Dataset import Dataset


NUM_CLASSES = 10
IMAGE_SIZE = 28  # Pixel width and height.
NUM_CHANNELS = 1 # grayscale

np.random.seed(133)
        
                        

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
          
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(count):
    initial = tf.constant(0.1, shape=[count])
    return tf.Variable(initial)
    
def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool(x, size=2):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                           strides=[1, size, size, 1], padding='SAME')

    
    
    
def tensorflow_conv2(train_dataset, train_labels, valid_dataset, valid_labels
                    ,test_dataset, test_labels, test_dataset_alt, test_labels_alt
                    ,save_summary=False, show_plot=False
                    ,batch_size=16, patch_size=5, depth=16
                    ,stride=1, pool_size=2
                    ,learn_rate=0.05, reg_param=0.001, keep_probability=0.5
                    ,num_steps=3001, full_layer_1=128):  
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
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name="x-input")
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, NUM_CLASSES), name="y-input")
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        tf_test_dataset_alt = tf.constant(test_dataset_alt)


        # Variables.
        # count the number of steps taken.
        global_step = tf.Variable(0, trainable=False)
        # input layer to conv layer 1
        W_1 = weight_variable([patch_size, patch_size, NUM_CHANNELS, depth])
        b_1 = bias_variable(depth) 
        # conv layer 1 to conv layer 2
        W_2 = weight_variable([patch_size, patch_size, depth, depth * 2])
        b_2 = bias_variable(depth * 2)
        # conv layer 2 to fully connected layer 3
        W_3 = weight_variable([(IMAGE_SIZE // (pool_size ** 2)) * (IMAGE_SIZE // (pool_size ** 2)) * (depth * 2), full_layer_1])
        b_3 = tf.Variable(tf.constant(1.0, shape=[full_layer_1]))
        # conv layer 2 to fully connected layer 3
        #W_4 = weight_variable([full_layer_1, full_layer_2])
        #b_4 = tf.Variable(tf.constant(1.0, shape=[full_layer_2]))
        # dropout layer
        #keep_prob = tf.placeholder(tf.float32)
        # output layer
        W_o = weight_variable([full_layer_1, NUM_CLASSES])
        b_o = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES]))
        
        
        if save_summary:
            tf.histogram_summary("weights-layer1", W_1)
            tf.histogram_summary("biases-layer1", b_1)
            tf.histogram_summary("weights-layer2", W_2)
            tf.histogram_summary("biases-layer2", b_2)
            tf.histogram_summary("weights-layer3", W_3)
            tf.histogram_summary("biases-layer3", b_3)
            #tf.histogram_summary("weights-layer4", W_4)
            #tf.histogram_summary("biases-layer4", b_4)
            tf.histogram_summary("weights-layer-output", W_o)
            tf.histogram_summary("biases-layer-output", b_o)
            #y_hist = tf.histogram_summary("y", y)
        
        
        # Model.
        def model(data, keep_prob=keep_probability):
            # convolutional layer 1
            activ = tf.nn.relu(conv2d(data, W_1, stride) + b_1)
            pool = max_pool(activ, pool_size)
            # convolutional layer 2
            activ = tf.nn.relu(conv2d(pool, W_2, stride) + b_2)
            pool = max_pool(activ, pool_size)
            # fully connected layer 3
            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
            activ = tf.nn.relu(tf.matmul(reshape, W_3) + b_3)
            # fully connected layer 4            
            #activ = tf.nn.relu(tf.matmul(activ, W_4) + b_4)
            # dropout layer
            activ = tf.nn.dropout(activ, keep_prob)
            # output layer
            return tf.matmul(activ, W_o) + b_o
        
        
        with tf.name_scope("loss-scope") as scope:            
            # Training computation.            
            logits = model(tf_train_dataset)
            reg = reg_param * (tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2) + tf.nn.l2_loss(W_3) + tf.nn.l2_loss(W_o))
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + reg
            
            if save_summary:
                tf.scalar_summary("loss-function", loss)            
            
        with tf.name_scope("loss-scope") as scope:            
            # Optimizer.            
            #learning_rate = tf.train.exponential_decay(learn_rate, global_step, 5000, 0.90, staircase=True)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
            
            optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)
            
        
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, keep_prob=1))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, keep_prob=1))
        test_prediction_alt = tf.nn.softmax(model(tf_test_dataset_alt, keep_prob=1))
        
        
        error_hist = np.array([])
        summary_step_size = np.max([num_steps // 100, 50])
        
        with tf.Session(graph=graph) as session:      
            start = time.time()                
                
            if save_summary:
                merged = tf.merge_all_summaries()
                writer = tf.train.SummaryWriter("./tensorboard/nn", session.graph_def)
            
            # initialize all the variables            
            tf.initialize_all_variables().run()
            
            # train over the configured number of steps
            for step in range(num_steps):
                
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                feed_dict = {
                    tf_train_dataset : batch_data, 
                    tf_train_labels : batch_labels
                }
                                
                if save_summary:    
                    _, l, predictions, summary_str = \
                        session.run([optimizer, loss, train_prediction, merged], feed_dict=feed_dict)
                else:
                    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                
                if (step % summary_step_size == 0):
                    current_accuracy = accuracy(valid_prediction.eval(), valid_labels)
                    error_hist = np.append(error_hist, 100-current_accuracy)
                    
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.2f%%' % accuracy(predictions, batch_labels))
                    print('Validation accuracy: %.2f%%\n' % current_accuracy)
                    
                if save_summary:
                    writer.add_summary(summary_str, step)
                    
            end = time.time()   
            
            #del train_dataset, train_labels, valid_dataset, valid_labels
            print('Test accuracy: %.2f%% after %.2f sec\n' 
                % (accuracy(test_prediction.eval(), test_labels), (end - start)))
            print('Test accuracy alt: %.2f%% after %.2f sec\n' 
                % (accuracy(test_prediction_alt.eval(), test_labels_alt), (end - start)))
                
            print ('Learning rate: ', learn_rate)
            print ('Keep probability: ', keep_probability)
            print ('Regularization rate: ', reg_param)
            print ('Batch size: ', batch_size)
            print ('Fully connected layer 1 size: ', full_layer_1)
                
            if show_plot:
                plt.plot(error_hist)
                plt.ylabel('Error rates')
                plt.show()

    
    
    
def tensorflow_conv1(train_dataset, train_labels, valid_dataset, valid_labels
                    ,test_dataset, test_labels, test_dataset_alt, test_labels_alt
                    ,save_summary=False, show_plot=False
                    ,batch_size=16, patch_size=5, depth=16
                    ,stride=1, pool_size=2
                    ,learn_rate=0.05, reg_param=0.001, keep_probability=0.5
                    ,num_steps=3001, full_layer_1=128):  
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
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name="x-input")
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, NUM_CLASSES), name="y-input")
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        tf_test_dataset_alt = tf.constant(test_dataset_alt)


        # Variables.
        # count the number of steps taken.
        global_step = tf.Variable(0, trainable=False)
        # input layer to conv layer 1
        W_1 = weight_variable([patch_size, patch_size, NUM_CHANNELS, depth])
        b_1 = bias_variable(depth) 
        # conv layer 2 to fully connected layer 3
        W_2 = weight_variable([IMAGE_SIZE * IMAGE_SIZE // (pool_size ** 4) * depth, full_layer_1])
        b_2 = tf.Variable(tf.constant(0.1, shape=[full_layer_1]))

        # output layer
        W_o = weight_variable([full_layer_1, NUM_CLASSES])
        b_o = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
        
        
        if save_summary:
            tf.histogram_summary("weights-layer1", W_1)
            tf.histogram_summary("biases-layer1", b_1)
            tf.histogram_summary("weights-layer2", W_2)
            tf.histogram_summary("biases-layer2", b_2)
            tf.histogram_summary("weights-layer-output", W_o)
            tf.histogram_summary("biases-layer-output", b_o)
            #y_hist = tf.histogram_summary("y", y)
        
        
        # Model.
        def model(data, keep_prob=keep_probability):
            # convolutional layer 1
            activ = tf.nn.relu(conv2d(data, W_1, stride) + b_1)
            pool = max_pool(activ, pool_size)
            # fully connected layer 3
            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
            activ = tf.nn.relu(tf.matmul(reshape, W_2) + b_2)
            # fully connected layer 4            
            #activ = tf.nn.relu(tf.matmul(activ, W_4) + b_4)
            # dropout layer
            activ = tf.nn.dropout(activ, keep_prob)
            # output layer
            return tf.matmul(activ, W_o) + b_o
        
        
        with tf.name_scope("loss-scope") as scope:            
            # Training computation.            
            logits = model(tf_train_dataset)
            reg = reg_param * (tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2) + tf.nn.l2_loss(W_o))
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + reg
            
            if save_summary:
                tf.scalar_summary("loss-function", loss)            
            
        with tf.name_scope("loss-scope") as scope:            
            # Optimizer.
            #learning_rate = tf.train.exponential_decay(learn_rate, global_step, 5000, 0.90, staircase=True)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        
            optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)
            
        
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, keep_prob=1))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, keep_prob=1))
        test_prediction_alt = tf.nn.softmax(model(tf_test_dataset_alt, keep_prob=1))
        
        
        error_hist = np.array([])
        summary_step_size = np.max([num_steps // 100, 50])
        
        with tf.Session(graph=graph) as session:      
            start = time.time()                
                
            if save_summary:
                merged = tf.merge_all_summaries()
                writer = tf.train.SummaryWriter("./tensorboard/nn", session.graph_def)
            
            # initialize all the variables            
            tf.initialize_all_variables().run()
            
            # train over the configured number of steps
            for step in range(num_steps):
                
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                feed_dict = {
                    tf_train_dataset : batch_data, 
                    tf_train_labels : batch_labels
                }
                                
                if save_summary:    
                    _, l, predictions, summary_str = \
                        session.run([optimizer, loss, train_prediction, merged], feed_dict=feed_dict)
                else:
                    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                
                if (step % summary_step_size == 0):
                    current_accuracy = accuracy(valid_prediction.eval(), valid_labels)
                    error_hist = np.append(error_hist, 100-current_accuracy)
                    
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.2f%%' % accuracy(predictions, batch_labels))
                    print('Validation accuracy: %.2f%%\n' % current_accuracy)
                    
                if save_summary:
                    writer.add_summary(summary_str, step)
                    
            end = time.time()   
            
            #del train_dataset, train_labels, valid_dataset, valid_labels
            print('Test accuracy: %.2f%% after %.2f sec\n' 
                % (accuracy(test_prediction.eval(), test_labels), (end - start)))
            print('Test accuracy alt: %.2f%% after %.2f sec\n' 
                % (accuracy(test_prediction_alt.eval(), test_labels_alt), (end - start)))
                
            print ('Learning rate: ', learn_rate)
            print ('Keep probability: ', keep_probability)
            print ('Regularization rate: ', reg_param)
            print ('Batch size: ', batch_size)
            print ('Fully connected layer 1 size: ', full_layer_1)
                
            if show_plot:
                plt.plot(error_hist)
                plt.ylabel('Error rates')
                plt.show()
            
            
    
                
if __name__ == '__main__':
    print ('Loading the dataset... ')
    start = time.time()
    data = Dataset('notMNIST', reformatted=True, verbose=True)
    train_dataset, train_labels, \
            valid_dataset, valid_labels, \
            test_dataset, test_labels = data.load()
    test_dataset_alt, test_labels_alt = data.load_test()
    end = time.time() 
    print ('Loading the dataset took %.2f sec.\n' % (end - start))
    
    tensorflow_conv2(train_dataset, train_labels, valid_dataset, valid_labels
                    ,test_dataset, test_labels, test_dataset_alt, test_labels_alt
                    ,batch_size=16, patch_size=5, depth=16, save_summary=False
                    ,learn_rate=0.002, keep_probability=0.5, reg_param=0.000#1
                    ,num_steps=501, full_layer_1=50)
        
    print ('Dataset: ', data.name)

