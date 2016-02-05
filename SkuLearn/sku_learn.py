# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:45:15 2015

@author: Radu
"""

import numpy as np
from sklearn import svm, decomposition
from sklearn.externals import joblib
import random
import os
#import scipy.sparse as sps
#from scipy.spatial.distance import cdist, pdist, squareform


# Constants
DATA_PATH = 'data.txt' #'alt_data.txt'
MODEL_SVM_PATH = 'model/sku_model.pkl'
MODEL_PCA_PATH = 'model/PCA_sku_model.pkl'
MODEL_NN_PATH = 'model/MLP_sku_model.pkl'
PCA_COMPONENTS = 50
MAX_SKU_LEN = 50
MAP_CHAR_LEN = 100
TRAIN_SET_SIZE = -1
RETRAIN_MODEL = True
STORE_MODEL = True


def get_mapping_table():
    '''
    Creates a mapping table containing each printable character
    '''
    import string
    return list(string.printable)


def translate(in_str, all_chars):
    '''
    Translates the input string into an array where each line is a character
    in the input string and each collumn is a character in the mapping table.
    The array has a value of 1 for each character in the SKU on the corresponding
    index of the mapping table and zero elsewhere.
    
    The array is reshaped into a vector.
    '''
    shape = (MAX_SKU_LEN, MAP_CHAR_LEN)
    out_arr = -1 * np.ones(shape, dtype=np.int)
    for i, c in enumerate(in_str):
        if (c in all_chars):
            out_arr[i, all_chars.index(c)] = 1
    
    return np.reshape(out_arr, -1)
    
    
def load_data():
    '''
    Loads data from the disk and builds the model
    '''
    # get mapping table
    all_chars = get_mapping_table()
    
    # read all SKUs from the list
    with open(DATA_PATH) as f:
        all_skus = f.read().splitlines()
    
    # translate each SKU to the corresponding input vector
    all_data = [translate(s, all_chars) for s in all_skus]
    
    if TRAIN_SET_SIZE != -1:
        all_data = random.sample(all_data, TRAIN_SET_SIZE)    
    
    return np.array(all_data), all_chars
    
    
def load_model(path):
    return joblib.load(path)        
    
    
def pca_decompose(all_data, size):
    print "fitting PCA..."
    pca = decomposition.PCA(n_components=size)
    pca.fit(all_data)
    print "done."
    print pca.noise_variance_ 
    print(pca.explained_variance_ratio_)
    
    # saves computed model to disk
    if STORE_MODEL:
        joblib.dump(pca, MODEL_PCA_PATH, compress=3)
        
    return pca
    

def train_model(train_set):
    '''
    Trains a one-class SVM and saves the model to disk
    '''    
    print "training SVM..."
    clf = svm.OneClassSVM()#(nu=0.99, kernel="rbf", gamma='auto')
    clf.fit(train_set)
    print "done."
    
    # saves computed model to disk
    if STORE_MODEL:
        joblib.dump(clf, MODEL_SVM_PATH, compress=3)

    return clf
    
    
def predict(clf, pca, value):
    all_chars = get_mapping_table()
    pred = np.array(translate(value, all_chars)).reshape(1, -1)
    X = pca.transform(pred)
    print "predicted value for ""%s"": %d" % (value, clf.predict(X)[0])
    

if __name__ == '__main__':
    # clear screen
    clear = lambda: os.system('cls')    
    clear()     
    
    if RETRAIN_MODEL == False and os.path.exists(MODEL_SVM_PATH) and os.path.exists(MODEL_PCA_PATH):
        pca = load_model(MODEL_PCA_PATH)
        clf = load_model(MODEL_SVM_PATH)
        print "using trained model"
    else:
        print "building new model"    
        #train_set, valid_set, test_set = cPickle.load(all_data)
        train_set, all_chars = load_data()
        # do PCA decomposition to the max size of the SKU
        pca = pca_decompose(train_set, PCA_COMPONENTS) 
        train_set = pca.transform(train_set)
        # train model
        clf = train_model(train_set)
        
    
    # predict on a new example
    test_sku = 'zzzzzzzzzzzzz' #'1558 - WHTRED 3004 40.5'    
    predict(clf, pca, test_sku)
