# -*- coding: utf-8 -*-
"""
Neural Networks over MNIST

Created on Fri Jan 29 22:42:01 2016

@author: Radu
"""

from __future__ import print_function  # import print() function from Python 3
import cPickle as pickle
import gzip, os
import numpy as np
import theano
import theano.tensor as T
import time

MNIST_FILE = './../data/mnist.pkl.gz'


NUM_CLASSES = 10
IMAGE_SIZE = 28  # Pixel width and height.

np.random.seed(133)
        

def load_datasets(pickle_file, verbose=False):
    '''
    Loads the MNIST dataset from a pickle file
    '''
    # Load the dataset
    with gzip.open(pickle_file, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)
        train_dataset, train_labels = shared_dataset(train_set)
        valid_dataset, valid_labels = shared_dataset(valid_set)
        test_dataset, test_labels = shared_dataset(test_set)
        #del train_set, valid_set, test_set  # hint to help gc free up memory
    
    if verbose:
        print ('Training set', train_dataset.shape, train_labels.shape)
        print ('Validation set', valid_dataset.shape, valid_labels.shape)
        print ('Test set', test_dataset.shape, test_labels.shape)
        
        batch_size = 500    # size of the minibatch
        # accessing the third minibatch of the training set        
        print (train_dataset[2 * batch_size: 3 * batch_size])
        print (train_labels[2 * batch_size: 3 * batch_size])
        
    return train_dataset, train_labels, valid_dataset, valid_labels, \
        test_dataset, test_labels


def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')
        
        
                
        
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(NUM_CLASSES) == labels[:,None]).astype(np.float32)
    return dataset, labels   
    
    
if __name__ == '__main__':   
    pickle_file = MNIST_FILE     
    if os.path.exists(pickle_file):
        # load the existing data 
        print ('Loading data from Pickle file ', pickle_file)
        train_dataset, train_labels, \
            valid_dataset, valid_labels, \
            test_dataset, test_labels = load_datasets(pickle_file, verbose=True)
    else:
        print ('MNIST Pickle file could not be found')