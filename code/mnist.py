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

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist

def cnn(X, Y, testX, testY ):
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
    model.fit({'input': X}, {'target': Y}, n_epoch=10,
               validation_set=({'input': testX}, {'target': testY}),
               snapshot_step=100, show_metric=True, run_id='convnet_mnist')

def do_dnn_1d(x_train, y_train,x_test , y_test):
    print "DNN and 1d"

    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (15, 10),
                        random_state = 1)
    print  clf
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    print "Hello MNIST"
    #X, Y, testX, testY = mnist.load_data(one_hot=True)
    X, Y, testX, testY = mnist.load_data(one_hot=False)
    #X = X.reshape([-1, 28, 28, 1])
    #testX = testX.reshape([-1, 28, 28, 1])

    do_dnn_1d(X, Y, testX, testY)