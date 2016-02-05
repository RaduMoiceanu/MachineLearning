# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import tarfile
import urllib
#from IPython.display import display, Image
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import cPickle as pickle
import gzip
import time


url = 'http://yaroslavvb.com/upload/notMNIST/'
pickle_file = 'notMNIST.pickle'
pickle_test = 'notMNIST-test.pickle'

num_classes = 10
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

np.random.seed(133)


def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print 'Found and verified', filename
  else:
    raise Exception(
      'Failed to verify' + filename + '. Can you get to it with a browser?')
  return filename


def extract(filename):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    
    if not os.path.exists(root):        
        tar = tarfile.open(filename)
        tar.extractall()
        tar.close()
        
    data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))]
    if len(data_folders) != num_classes:
        raise Exception('Expected %d folders, one per class. Found %d instead.' % (num_classes, len(data_folders)))
        
    #print data_folders
    return data_folders


def load(data_folders, min_num_images, max_num_images):
  dataset = np.ndarray(
    shape=(max_num_images, image_size, image_size), dtype=np.float32)
  labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
  label_index = 0
  image_index = 0
  for folder in data_folders:
    print folder
    for image in os.listdir(folder):
      if image_index >= max_num_images:
        raise Exception('More images than expected: %d >= %d' % (image_index, max_num_images))
      image_file = os.path.join(folder, image)
      try:
        image_data = (ndimage.imread(image_file).astype(float) -
                      pixel_depth / 2) / pixel_depth
        if image_data.shape != (image_size, image_size):
          raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset[image_index, :, :] = image_data
        labels[image_index] = label_index
        image_index += 1
      except IOError as e:
        print 'Could not read:', image_file, ':', e, '- it\'s ok, skipping.'
    label_index += 1
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  labels = labels[0:num_images]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' % (
        num_images, min_num_images))
  print 'Full dataset tensor:', dataset.shape
  print 'Mean:', np.mean(dataset)
  print 'Standard deviation:', np.std(dataset)
  print 'Labels:', labels.shape
  return dataset, labels
    
    
def display_rand_img(train_dataset, rand_img_count = 10):
    '''
    Displays a number of random images from the dataset
    '''
    big_img = Image.new('F', (rand_img_count * image_size, image_size))
    i = 0
    for idx in np.random.randint(0, train_dataset.shape[0], rand_img_count):
        img = Image.fromarray(train_dataset[idx] * pixel_depth + pixel_depth / 2)
        big_img.paste(img, (i * image_size, 0))
        i = i+1
    big_img.show()


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def store_datasets(pickle_file, train_dataset, train_labels
                    ,valid_dataset, valid_labels
                    ,test_dataset, test_labels):
    try:
      f = open(pickle_file, 'wb')
      save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
      }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print 'Unable to save data to', pickle_file, ':', e
      raise

      
def store_dataset_test(pickle_file, test_dataset, test_labels):
    try:
      #with gzip.GzipFile(pickle_file, 'wb') as f:
      f = open(pickle_file, 'wb')
      save = {
        'test_dataset': test_dataset,
        'test_labels': test_labels,
      }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print 'Unable to save data to', pickle_file, ':', e
      raise


def load_datasets(pickle_file):
    try:
        data = pickle.load(open(pickle_file, "rb"))
        return data["train_dataset"], data["train_labels"], \
               data["valid_dataset"], data["valid_labels"], \
               data["test_dataset"], data["test_labels"]
    except Exception as e:
        print 'Unable to load data from ', pickle_file, ':', e
        raise
    

def linear_regr(train_dataset, train_labels, test_dataset, test_labels): 
    X_train = train_dataset.reshape((-1, train_dataset.shape[1] * train_dataset.shape[2]))
    X_test = test_dataset.reshape((-1, test_dataset.shape[1] * test_dataset.shape[2]))
     
    # Create linear regression object
    logreg = LogisticRegression()
    
    # fit the training data
    start = time.time()
    logreg.fit(X_train, train_labels)
    end = time.time()    
    
    # predict over the test data
    predict_labels = logreg.predict(X_test)
    accuracy = accuracy_score(test_labels, predict_labels)
    
    print('Fitting took: %.2f sec for %d rows resulting in %.2f %% accuracy.' % 
         ((end - start), X_train.shape[0], accuracy * 100))
    
    
def play_linear_regr(train_dataset, train_labels, test_dataset, test_labels):
    '''
    Play with different train dataset sizes and observe the effect on accuracy
    '''
    train_size = 50
    linear_regr(train_dataset[:train_size,:,:], train_labels[:train_size], \
                test_dataset, test_labels)
                
    train_size = 100
    linear_regr(train_dataset[:train_size,:,:], train_labels[:train_size], \
                test_dataset, test_labels)
                
    train_size = 1000
    linear_regr(train_dataset[:train_size,:,:], train_labels[:train_size], \
                test_dataset, test_labels)
                
    train_size = 5000
    linear_regr(train_dataset[:train_size,:,:], train_labels[:train_size], \
                test_dataset, test_labels)
                
    train_size = 10000
    linear_regr(train_dataset[:train_size,:,:], train_labels[:train_size], \
                test_dataset, test_labels)
    
    
    
if __name__ == '__main__':    
    if not os.path.exists(pickle_file):
        print 'Pickle file not found ', pickle_file
        
        # get dataset, if needed
        train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
        #test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
      
        # extract images, if needed
        train_folders = extract(train_filename)
        #test_folders = extract(test_filename)
      
        # load data into memory
        train_dataset, train_labels = load(train_folders, 450000, 550000)
        #test_dataset, test_labels = load(test_folders, 18000, 20000)
    
        # maybe randomisze it
        train_dataset, train_labels = randomize(train_dataset, train_labels)
        #test_dataset, test_labels = randomize(test_dataset, test_labels)
    
        test_size = 20000
        valid_size = 10000
        train_size = -1 #200000
        
        # extract a subset of data and a validation dataset
        test_dataset = train_dataset[:test_size,:,:]
        test_labels = train_labels[:test_size]
        valid_dataset = train_dataset[test_size:valid_size+test_size,:,:]
        valid_labels = train_labels[test_size:valid_size+test_size]
        train_dataset = train_dataset[valid_size+test_size:-1,:,:]
        train_labels = train_labels[valid_size+test_size:-1]
        print 'Training', train_dataset.shape, train_labels.shape
        print 'Validation', valid_dataset.shape, valid_labels.shape      
      
        # save the data for later reuse
        store_datasets(pickle_file, train_dataset, train_labels, valid_dataset, 
                       valid_labels, test_dataset, test_labels)
        #store_dataset_test(pickle_test, test_dataset, test_labels)
        statinfo = os.stat(pickle_file)
        print 'Compressed pickle size:', statinfo.st_size
    else:
        # load the existing data 
        print 'Loading data from Pickle file ', pickle_file
        train_dataset, train_labels, \
            valid_dataset, valid_labels, \
            test_dataset, test_labels = load_datasets(pickle_file)
        
    #display_rand_img(test_dataset, 20)
    #play_linear_regr(train_dataset, train_labels, valid_dataset, valid_labels)

# see how many duplicates we have
#train_valid_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
#train_test_dataset = np.concatenate((train_dataset, test_dataset), axis=0)