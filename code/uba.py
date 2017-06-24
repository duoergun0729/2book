
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

    transformer = TfidfTransformer(smooth_idf=False)
    transformer.fit(x)
    x_test = transformer.transform(x_test)
    x_train = transformer.transform(x_train)

    return x_train, x_test, y_train, y_test


def get_features_by_ngram():
    global max_features
    global  index
    x_arr,y=get_cmdlines()
    x=[]

    for i,v in enumerate(x_arr):
        v=" ".join(v)
        x.append(v)

    vectorizer = CountVectorizer(
                                 ngram_range=(2, 4),
                                 token_pattern=r'\b\w+\b',
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

    transformer = TfidfTransformer(smooth_idf=False)
    transformer.fit(x)
    x_test = transformer.transform(x_test)
    x_train = transformer.transform(x_train)

    return x_train, x_test, y_train, y_test

def  get_features_by_wordseq():
    global max_features
    global  index
    x_arr,y=get_cmdlines()
    x=[]

    for i,v in enumerate(x_arr):
        v=" ".join(v)
        x.append(v)

    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=100,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x=vp.fit_transform(x, unused_y=None)
    x = np.array(list(x))

    x_train = x[0:index, ]
    x_test = x[index:, ]
    y_train = y[0:index, ]
    y_test = y[index:, ]

    #x_train = vp.transform(x_train)
    #x_train = np.array(list(x_train))
    #x_test = vp.transform(x_test)
    #x_test = np.array(list(x_test))


    return x_train, x_test, y_train, y_test

def do_xgboost(x_train, x_test, y_train, y_test):
    print "xgboost"
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)


def do_rnn_wordbag(trainX, testX, trainY, testY):
    y_test=testY
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, 100])
    net = tflearn.embedding(net, input_dim=100, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=10,run_id="sms",n_epoch=5)

    y_predict_list = model.predict(testX)
    print y_predict_list

    y_predict = []
    for i in y_predict_list:
        print  i[0]
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    print(classification_report(y_test, y_predict))
    print metrics.confusion_matrix(y_test, y_predict)

if __name__ == "__main__":
    print "Hello uba"

    print "xgboost and wordbag"
    max_features=100
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_wordbag()
    do_xgboost(x_train, x_test, y_train, y_test)


    print "xgboost and ngram"
    max_features=1000
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_ngram()
    do_xgboost(x_train, x_test, y_train, y_test)

    print "xgboost and wordseq"
    max_features=1000
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_wordseq()
    do_xgboost(x_train, x_test, y_train, y_test)

    print "rnn and wordseq"
    max_features=100
    print "max_features=%d" % max_features
    x_train, x_test, y_train, y_test=get_features_by_wordseq()
    do_rnn_wordbag(x_train, x_test, y_train, y_test)