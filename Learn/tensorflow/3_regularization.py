# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 16:04:16 2016

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
    dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(NUM_CLASSES) == labels[:,None]).astype(np.float32)
    return dataset, labels   
                        

def accuracy(predictions, labels):
    correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1))
    return (100.0 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) 
                    
                      

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(count):
    return tf.Variable(tf.zeros([count])) 
    
def evaluate(tensor, weights_0, biases_0, weights_1, biases_1, weights_2, biases_2):
    activations_1 = tf.nn.relu(tf.matmul(tensor, weights_0) + biases_0)
    activations_2 = tf.nn.relu(tf.matmul(activations_1, weights_1) + biases_1)
    return tf.nn.softmax(tf.matmul(activations_2, weights_2) + biases_2)


def tensorflow_nn_dropout(train_dataset, train_labels, valid_dataset, valid_labels
                          ,test_dataset, test_labels, save_best_params=False
                          ,num_steps = 3001, batch_size = 128
                          ,hidden_count=1024, hidden_count_l2=512
                          ,learn_rate=0.01, reg_param=0.001, keep_probability=0.5):  
    '''
    Tensorflow implementation of classification using NN of a single hidden layer
    and Stochastic Gradient Descent (SGD)
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

        # input layer to hidden layer
        weights_0 = weight_variable([IMAGE_SIZE * IMAGE_SIZE, hidden_count])
        biases_0 = bias_variable(hidden_count) 
        # hidden layer to output layer
        weights_1 = weight_variable([hidden_count, hidden_count_l2])
        biases_1 = bias_variable(hidden_count_l2)
        
        weights_2 = weight_variable([hidden_count_l2, NUM_CLASSES])
        biases_2 = bias_variable(NUM_CLASSES)
       
       # probability that a neuron's output is kept during dropout     
        keep_prob = tf.placeholder("float")
        # count the number of steps taken.
        global_step = tf.Variable(0, trainable=False)
        # best accuracy on the validation set
        best_validation = 0.0        
        
        
        # Training computation.
        # ---------------------
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        activations_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_0) + biases_0)
        #logits = tf.matmul(activations_1, weights_1) + biases_1
        
        activations_2 = tf.nn.relu(tf.matmul(activations_1, weights_1) + biases_1)
        #logits = tf.matmul(activations_1, weights_1) + biases_1
        
        activations_2_drop = tf.nn.dropout(activations_2, keep_probability)
        logits_drop = tf.matmul(activations_2_drop, weights_2) + biases_2
        
        reg = reg_param * (tf.nn.l2_loss(weights_0) + tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_drop, tf_train_labels)) + reg
        
        
        if save_best_params:
            # Add summary ops to collect data
            weights_0_hist = tf.histogram_summary("weights_0", weights_0)
            biases_0_hist = tf.histogram_summary("biases_0", biases_0)
            weights_1_hist = tf.histogram_summary("weights_1", weights_1)
            biases_1_hist = tf.histogram_summary("biases_1", biases_1)
            loss_hist = tf.histogram_summary("loss", loss)  
            
            # Create a saver.
            saver = tf.train.Saver()   
            
        
        # Optimizer.
        # ----------
        # We are going to find the minimum of this loss using gradient descent.
        
        #optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
        
        learning_rate = tf.train.exponential_decay(learn_rate, global_step, 5000, 0.90, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=learn_rate, decay=0.9).minimize(loss)
             
        # AdaGrad learns very quickly on MNIST compared to regular SGD, 
        # but tends to slightly overfit the training batches (accuracy of 100%) 
        # while producing a slightly lower accuracy over the validation and test data (95% vs 96%).
        #
        # On notMNIST it performs very poorly.
        #
        # What it does is it favors sparse features more so that their influence is
        # not discounted because they appear rarely in the training data
        # optimizer = tf.train.AdagradOptimizer(learning_rate=learn_rate).minimize(loss)
             
        #optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.75).minimize(loss)
        
        
        error_hist = np.array([])
        
        # Predictions for the training, validation, and test data.
        # --------------------------------------------------------
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits_drop)
        valid_prediction = evaluate(tf_valid_dataset, weights_0, biases_0, weights_1, biases_1, weights_2, biases_2)
        test_prediction = evaluate(tf_test_dataset, weights_0, biases_0, weights_1, biases_1, weights_2, biases_2)
                
                
        # you can run the operations on this graph as many times as you want by 
        # calling session.run(), providing it outputs to fetch from the graph that get returned.
        with tf.Session(graph=graph) as session:            
            start = time.time()
            
            if save_best_params:
                merged = tf.merge_all_summaries()
                writer = tf.train.SummaryWriter("./tensorboard/nn", session.graph_def)
        
            # This is a one-time operation which ensures the parameters get initialized as
            # we described in the graph: random weights for the matrix, zeros for the
            # biases. 
            tf.initialize_all_variables().run()
            
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
                    tf_train_labels : batch_labels,
                    keep_prob: keep_probability
                }
                    
                    
                # Run the computations. 
                # ---------------------
                # We tell .run() that we want to run the optimizer and get the loss value 
                # and the training predictions returned as numpy arrays.
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                             
                if (step % 50 == 0):
                    current_accuracy = accuracy(predictions, batch_labels).eval()
                    error_hist = np.append(error_hist, 100-current_accuracy)
                    if save_best_params and best_validation < current_accuracy:
                        best_validation = current_accuracy
                        saver.save(session, '/home/radu/Documents/tensorboard/nn', global_step=step, latest_filename='nn-last')
                
                if (step % 500 == 0):                
                    print ("Minibatch loss at step", step, ":", l)
                    print ("Minibatch accuracy: %.1f%%" % current_accuracy)
                    print ("Validation accuracy: %.1f%% \n" % \
                        accuracy(valid_prediction.eval(), valid_labels).eval())
                
            end = time.time()   
            print ("Final train accuracy: %.1f%%" % accuracy(predictions, batch_labels).eval())
            print ("Validation accuracy: %.1f%% \n" % accuracy(valid_prediction.eval(), valid_labels).eval())
            print ("Test accuracy: %.1f%% after %.2f sec\n" % \
                (accuracy(test_prediction.eval(), test_labels).eval(), (end - start)))
                
            print ('Learning rate: ', learn_rate)
            print ('Regularization rate: ', reg_param)
            print ('Keep probability: ', keep_probability)
            print ('Batch size: ', batch_size)
            print ('Hidden layer 1 size: ', hidden_count)
            print ('Hidden layer 2 size: ', hidden_count_l2)
                
            plt.plot(error_hist)
            plt.ylabel('Error rates')
            plt.show()
                
            #best_params = saver.restore(session, 'tensorflow_nn_dropout-last')
            #print "Test accuracy with early termination: %.1f%% \n" % \
            #    accuracy(test_prediction.eval(), test_labels).eval()
    
                
if __name__ == '__main__':
    # load the existing data 
    train_dataset, train_labels, \
        valid_dataset, valid_labels, \
        test_dataset, test_labels = load_datasets_notMNIST(pickle_file)
        #test_dataset, test_labels = load_datasets_MNIST(MNIST_FILE, verbose=True)
        
    # reformat the data to a flat matrix and the labels as 1-hot encoding
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    
    tensorflow_nn_dropout(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, 
                          num_steps=20001, batch_size=64, hidden_count = 1024,
                          learn_rate=0.03, reg_param=0.005, keep_probability=1.) 
    
    