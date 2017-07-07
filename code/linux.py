# -*- coding:utf-8 -*-

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


def load_all_files():
    import glob
    x=[]
    y=[]
    #加载攻击样本
    files=glob.glob("../data/rootkit/ADFA-LD/Attack_Data_Master/*/*")
    for file in files:
        with open(file) as f:
            lines=f.readlines()
        x.append(" ".join(lines))
        y.append(1)
    print "Load black data %d" % len(x)
    #加载正常样本
    files=glob.glob("../data/rootkit/ADFA-LD/Training_Data_Master/*")
    for file in files:
        with open(file) as f:
            lines=f.readlines()
        x.append(" ".join(lines))
        y.append(0)
    print "Load full data %d" % len(x)

    return x,y

def do_mlp(x_train, x_test, y_train, y_test):
    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)

def do_xgboost(x_train, x_test, y_train, y_test):
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)

def do_nb(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)

def get_feature_wordbag():
    max_features=1000
    x,y=load_all_files()
    vectorizer = CountVectorizer(
                                 ngram_range=(3, 3),
                                 token_pattern=r'\b\d+\b',
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print vectorizer
    x = vectorizer.fit_transform(x)

    transformer = TfidfTransformer(smooth_idf=False)
    x=transformer.fit_transform(x)

    x = x.toarray()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    print "Hello Linux Rootkit"
    print "3-Gram&tf-idf and nb"
    x_train, x_test, y_train, y_test=get_feature_wordbag()
    do_nb(x_train, x_test, y_train, y_test)
    print "2-Gram&tf-idf and xgboost"
    do_xgboost(x_train, x_test, y_train, y_test)
    print "2-Gram&tf-idf and mlp"
    do_mlp(x_train, x_test, y_train, y_test)

