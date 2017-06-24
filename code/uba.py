
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.neural_network import MLPClassifier
from tflearn.layers.normalization import local_response_normalization
from tensorflow.contrib import learn
import gensim
import re
from collections import namedtuple
from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from random import shuffle
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn import preprocessing

cmdlines_file="../data/uba/MasqueradeDat/User7"
labels_file="../data/uba/MasqueradeDat/label.txt"
max_features=100
index = 80

def get_cmdlines():
    x=np.loadtxt(cmdlines_file,dtype=str)
    x=x.reshape((150,100))
    y=np.loadtxt(labels_file, dtype=int,usecols=6)
    y=y.reshape((100, 1))
    y_train=np.zeros([50,1],int)
    y=np.concatenate([y_train,y])
    y=y.reshape((150, ))

    return x,y

def get_features_by_wordbag():
    global max_features
    global  index
    x_arr,y=get_cmdlines()
    x=[]

    for i,v in enumerate(x_arr):
        v=" ".join(v)
        print v
        x.append(v)

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    x=vectorizer.fit_transform(x)

    x_train=x[0:index,]
    x_test=x[index:,]
    y_train=y[0:index,]
    y_test=y[index:,]
    print y_train
    print y_test

    return x_train, x_test, y_train, y_test

def do_xgboost(x_train, x_test, y_train, y_test):
    print "xgboost"
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    print "Hello uba"

    print "xgboost and wordbag"
    x_train, x_test, y_train, y_test=get_features_by_wordbag()
    do_xgboost(x_train, x_test, y_train, y_test)


