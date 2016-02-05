# -*- coding: utf-8 -*-
"""
Handles the learning datasets - currently MNIST and notMNIST.

Created on Thu Feb  4 17:05:27 2016

@author: radu
"""
from __future__ import print_function
import numpy as np
import cPickle as pickle
import gzip, os, tarfile, urllib
from scipy import ndimage


class Dataset:
    notMNIST_url = 'http://yaroslavvb.com/upload/notMNIST/'
    notMNIST_pickle = 'notMNIST.pkl.gz'
    notMNIST_pickle_test = 'notMNIST-test.pkl.gz'
    
    NUM_CLASSES = 10
    IMAGE_SIZE = 28  # Pixel width and height.
    NUM_CHANNELS = 1 # grayscale
    PIXEL_DEPTH = 255.0  # Number of levels per pixel.
    RANDOM_SEED = 133
    
    
    def __init__(self, name='MNIST', reformatted=True, verbose=False):
        self.name = name
        self.reformatted = reformatted
        self.verbose = verbose
        
        
    def load(self):
        '''
        Loads and returns the dataset specified by the "name" parameter 
        of the constructor
        '''
        if self.name == 'MNIST':
            return self.MNIST()
        elif self.name == 'notMNIST':
            return self.notMNIST()
        
        
    def load_test(self):
        '''
        Loads and returns the test dataset specified by the "name" parameter 
        of the constructor
        '''
        if self.name == 'MNIST':
            return self.MNIST_test()
        elif self.name == 'notMNIST':
            return self.notMNIST_test()
        
    
    def notMNIST(self, pickle_file=notMNIST_pickle, pickle_file_test=notMNIST_pickle_test):
        '''
        Returns the "notMNIST" dataset
        '''        
        # check to see if pickle file exists
        if not os.path.exists(pickle_file):
            if self.verbose:
                print ('Pickle file not found ', pickle_file)            
            
            train_dataset, train_labels, valid_dataset, valid_labels, \
                test_dataset, test_labels = self._get_notMNIST()
        else:
            with gzip.open(pickle_file, 'rb') as f:
            #with open(pickle_file, 'rb') as f:
                save = pickle.load(f)
                train_dataset = save['train_dataset']
                train_labels = save['train_labels']
                valid_dataset = save['valid_dataset']
                valid_labels = save['valid_labels']
                test_dataset = save['test_dataset']
                test_labels = save['test_labels']
                del save  # hint to help gc free up memory
            
        # reformat, if needed
        if self.reformatted:                
            # reformat the data to a flat matrix and the labels as 1-hot encoding
            train_dataset, train_labels = self.reformat(train_dataset, train_labels)
            valid_dataset, valid_labels = self.reformat(valid_dataset, valid_labels)
            test_dataset, test_labels = self.reformat(test_dataset, test_labels)
        
        if self.verbose:
            print ('Training set', train_dataset.shape, train_labels.shape)
            print ('Validation set', valid_dataset.shape, valid_labels.shape)
            print ('Test set', test_dataset.shape, test_labels.shape)
                
        return train_dataset, train_labels, valid_dataset, valid_labels, \
            test_dataset, test_labels
    
    
    def notMNIST_test(self, pickle_file='notMNIST-test.pkl.gz'):
        '''
        Returns the "notMNIST" test dataset
        '''
        with gzip.open(pickle_file, 'rb') as f:
        #with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
            del save  # hint to help gc free up memory
            
        if self.reformatted:                
            test_dataset, test_labels = self.reformat(test_dataset, test_labels)
            
        if self.verbose:
            print ('Test set', test_dataset.shape, test_labels.shape)
                
        return test_dataset, test_labels
            
            
    def MNIST(self, pickle_file='mnist.pkl.gz'):
        '''
        Returns the MNIST dataset
        '''
        # Load the dataset
        with gzip.open(pickle_file, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
            train_dataset, train_labels = train_set[0], train_set[1]
            valid_dataset, valid_labels = valid_set[0], valid_set[1]
            test_dataset, test_labels = test_set[0], test_set[1]
            del train_set, valid_set, test_set  # hint to help gc free up memory
            
        if self.reformatted:                
            # reformat the data to a flat matrix and the labels as 1-hot encoding
            train_dataset, train_labels = self.reformat(train_dataset, train_labels)
            valid_dataset, valid_labels = self.reformat(valid_dataset, valid_labels)
            test_dataset, test_labels = self.reformat(test_dataset, test_labels)
        
        if self.verbose:
            print ('Training set', train_dataset.shape, train_labels.shape)
            print ('Validation set', valid_dataset.shape, valid_labels.shape)
            print ('Test set', test_dataset.shape, test_labels.shape)
            
        return train_dataset, train_labels, valid_dataset, valid_labels, \
            test_dataset, test_labels
            
            
    def MNIST_test(self, pickle_file='mnist.pkl.gz'):
        '''
        Returns the MNIST test dataset
        '''
        # Load the dataset
        with gzip.open(pickle_file, 'rb') as f:
            _, _, test_set = pickle.load(f)
            test_dataset, test_labels = test_set[0], test_set[1]
            del test_set  # hint to help gc free up memory
            
        if self.reformatted:                
            test_dataset, test_labels = self.reformat(test_dataset, test_labels)
        
        if self.verbose:
            print ('Test set', test_dataset.shape, test_labels.shape)
            
        return test_dataset, test_labels
                    
            
    def reformat(self, dataset, labels):
        '''
        Reformats the data to a flat matrix and the labels as 1-hot encoding
        '''
        dataset = dataset.reshape((-1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(self.NUM_CLASSES) == labels[:,None]).astype(np.float32)
        return dataset, labels   
        
    
    
    def _get_notMNIST(self, test_size = 20000, valid_size = 10000, train_size = -1, #200000
                      archive_filename = 'notMNIST_large.tar.gz',
                      archive_filename_test = 'notMNIST_small.tar.gz',
                      pickle_file=notMNIST_pickle, 
                      pickle_file_test=notMNIST_pickle_test):
        '''
        Gets the "notMNIST" dataset and stores it as pickle files
        '''         
        tar_exists = os.path.exists(archive_filename)
        train_filename = self._maybe_download(archive_filename, self.notMNIST_url, 247336696)
        test_filename = self._maybe_download(archive_filename_test, self.notMNIST_url, 8458043)
        if not tar_exists:
            train_folders = self._extract(train_filename)
            test_folders = self._extract(test_filename)
        else:
            root = os.path.splitext(os.path.splitext(archive_filename)[0])[0]
            train_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))]
            
            root_test = os.path.splitext(os.path.splitext(archive_filename_test)[0])[0]
            test_folders = [os.path.join(root_test, d) for d in sorted(os.listdir(root_test))]
        
        train_dataset, train_labels = self._load_images(train_folders, 450000, 550000)
        test_dataset, test_labels = self._load_images(test_folders, 18000, 20000)
        
        train_dataset, train_labels = self._randomize(train_dataset, train_labels)
        test_dataset, test_labels = self._randomize(test_dataset, test_labels)
                 
    
        # extract a subset of data and a validation dataset
        test_dataset = train_dataset[:test_size, :, :]
        test_labels = train_labels[:test_size]
        valid_dataset = train_dataset[test_size : valid_size+test_size, :, :]
        valid_labels = train_labels[test_size : valid_size+test_size]
        train_dataset = train_dataset[valid_size+test_size : train_size, :, :]
        train_labels = train_labels[valid_size+test_size : train_size]
        
        if self.verbose:
            print ('Training', train_dataset.shape, train_labels.shape)
            print ('Validation', valid_dataset.shape, valid_labels.shape)
  
        # save the data for later reuse
        self._store_datasets(pickle_file, train_dataset, train_labels, valid_dataset, 
                             valid_labels, test_dataset, test_labels)
        self._store_dataset_test(pickle_file_test, test_dataset, test_labels)
        
        statinfo = os.stat(pickle_file)
        if self.verbose:
            print ('Compressed pickle size:', statinfo.st_size)
            
        return train_dataset, train_labels, valid_dataset, valid_labels, \
            test_dataset, test_labels
        
        
    def _maybe_download(self, filename, url, expected_bytes):
        '''
        Downloads a file if not present, and make sure it's the right size.
        '''
        if not os.path.exists(filename):
            filename, _ = urllib.urlretrieve(url + filename, filename)
        
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes and self.verbose:
            print ('Found and verified', filename)
        else:
            raise Exception('Failed to verify' + filename + '. Can you get to it with a browser?')
        
        return filename
        
        
    def _extract(self, filename):
        '''
        Extracts the data from a tar archive
        '''
        root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
        
        if not os.path.exists(root):        
            tar = tarfile.open(filename)
            tar.extractall()
            tar.close()
            
        data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))]
        if len(data_folders) != self.NUM_CLASSES:
            raise Exception('Expected %d folders, one per class. Found %d instead.' % (self.NUM_CLASSES, len(data_folders)))
            
        if self.verbose:
            print (data_folders)
            
        return data_folders
        
        
    def _load_images(self, data_folders, min_num_images, max_num_images):
        '''
        Loads the image data from the data folders to the memory
        '''
        dataset = np.ndarray(shape=(max_num_images, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.float32)
        labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
        label_index = 0
        image_index = 0
        for folder in data_folders:
            if self.verbose:
                print (folder)
            
            for image in os.listdir(folder):
                if image_index >= max_num_images:
                    raise Exception('More images than expected: %d >= %d' % (image_index, max_num_images))
                image_file = os.path.join(folder, image)
                
                try:
                    image_data = (ndimage.imread(image_file).astype(float) - self.PIXEL_DEPTH / 2) / self.PIXEL_DEPTH
                    if image_data.shape != (self.IMAGE_SIZE, self.IMAGE_SIZE):
                        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                    dataset[image_index, :, :] = image_data
                    labels[image_index] = label_index
                    image_index += 1
                except IOError as e:
                    print ('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
                
            label_index += 1
            
        num_images = image_index
        dataset = dataset[0:num_images, :, :]
        labels = labels[0:num_images]
        if num_images < min_num_images:
            raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

        if self.verbose:
            print ('Full dataset tensor:', dataset.shape)
            print ('Mean:', np.mean(dataset))
            print ('Standard deviation:', np.std(dataset))
            print ('Labels:', labels.shape)
        
        return dataset, labels
        
        
    def _randomize(self, dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation, :, :]
        shuffled_labels = labels[permutation]
        
        return shuffled_dataset, shuffled_labels


    def _store_datasets(self, pickle_file, train_dataset, train_labels
                        ,valid_dataset, valid_labels, test_dataset, test_labels):
        '''
        Stores the dataset as a pickle file
        '''
        try:
            #f = open(pickle_file, 'wb')
            with gzip.GzipFile(pickle_file, 'wb') as f:
                save = {
                    'train_dataset': train_dataset,
                    'train_labels': train_labels,
                    'valid_dataset': valid_dataset,
                    'valid_labels': valid_labels,
                    'test_dataset': test_dataset,
                    'test_labels': test_labels,
                }
                pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            #f.close()
        except Exception as e:
            print ('Unable to save data to', pickle_file, ':', e)
            raise

      
    def _store_dataset_test(self, pickle_file, test_dataset, test_labels):
        '''
        Stores the test dataset as a pickle file
        '''
        try:
            #f = open(pickle_file, 'wb')
            with gzip.GzipFile(pickle_file, 'wb') as f:
                save = {
                    'test_dataset': test_dataset,
                    'test_labels': test_labels,
                }
                pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            #f.close()
        except Exception as e:
            print ('Unable to save data to', pickle_file, ':', e)
            raise