# -*- coding: utf-8 -*-

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn import neighbors

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist

def do_cnn_2d(X, Y, testX, testY ):
    # Building convolutional network
    network = input_data(shape=[None, 28, 28, 1], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': X}, {'target': Y}, n_epoch=5,
               validation_set=({'input': testX}, {'target': testY}),
               snapshot_step=100, show_metric=True, run_id='mnist')

def do_dnn_1d(x_train, y_train,x_test , y_test):
    print "DNN and 1d"

    # Building deep neural network
    input_layer = tflearn.input_data(shape=[None, 784])
    dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout1 = tflearn.dropout(dense1, 0.8)
    dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout2 = tflearn.dropout(dense2, 0.8)
    softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')

    # Regression using SGD with learning rate decay and Top-3 accuracy
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    top_k = tflearn.metrics.Top_k(3)
    net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=10, validation_set=(testX, testY),
              show_metric=True, run_id="mnist")

def do_svm_1d(x_train, y_train,x_test, y_test):
    print "SVM and 1d"
    clf = svm.SVC(decision_function_shape='ovo')
    print clf
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    #print metrics.confusion_matrix(y_test, y_pred)

def do_knn_1d(x_train, y_train,x_test, y_test):
    print "KNN and 1d"
    clf = neighbors.KNeighborsClassifier(n_neighbors=15)
    print clf
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    #print metrics.confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    print "Hello MNIST"
    #X, Y, testX, testY = mnist.load_data(one_hot=False)
    #1d
    #print  testX
    #do_dnn_1d(X, Y, testX, testY)
    #do_svm_1d(X, Y, testX, testY)
    #do_knn_1d(X, Y, testX, testY)



    #2d
    X, Y, testX, testY = mnist.load_data(one_hot=True)
    X = X.reshape([-1, 28, 28, 1])
    testX = testX.reshape([-1, 28, 28, 1])

    #cnn
    do_cnn_2d(X, Y, testX, testY)
