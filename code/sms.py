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

max_features=1000
max_document_length=160
vocabulary=None
doc2ver_bin="smsdoc2ver.bin"
word2ver_bin="smsword2ver.bin"
SentimentDocument = namedtuple('SentimentDocument', 'words tags')




def load_all_files():
    x=[]
    y=[]
    datafile="../data/sms/smsspamcollection/SMSSpamCollection.txt"
    with open(datafile) as f:
        for line in f:
            line=line.strip('\n')
            label,text=line.split('\t')
            x.append(text)
            if label == 'ham':
                y.append(0)
            else:
                y.append(1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    return x_train, x_test, y_train, y_test


def get_features_by_wordbag():
    global max_features
    x_train, x_test, y_train, y_test=load_all_files()

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print vectorizer
    x_train=vectorizer.fit_transform(x_train)
    x_train=x_train.toarray()
    vocabulary=vectorizer.vocabulary_

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 vocabulary=vocabulary,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print vectorizer
    x_test=vectorizer.fit_transform(x_test)
    x_test=x_test.toarray()

    return x_train, x_test, y_train, y_test


def do_nb_wordbag(x_train, x_test, y_train, y_test):
    print "NB and wordbag"
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)

def do_nb_doc2vec(x_train, x_test, y_train, y_test):
    print "NB and doc2vec"
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

def do_nb_word2vec(x_train, x_test, y_train, y_test):
    print "NB and word2vec"
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))


def do_svm_wordbag(x_train, x_test, y_train, y_test):
    print "SVM and wordbag"
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)


def do_svm_doc2vec(x_train, x_test, y_train, y_test):
    print "SVM and doc2vec"
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

def do_svm_word2vec(x_train, x_test, y_train, y_test):
    print "SVM and word2vec"
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

def do_rf_doc2vec(x_train, x_test, y_train, y_test):
    print "rf and doc2vec"
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

def do_rf_word2vec(x_train, x_test, y_train, y_test):
    print "rf and word2vec"
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

def get_features_by_wordbag_tfidf():
    global max_features
    x_train, x_test, y_train, y_test=load_all_files()

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1,
                                 binary=True)
    print vectorizer
    x_train=vectorizer.fit_transform(x_train)
    x_train=x_train.toarray()
    vocabulary=vectorizer.vocabulary_

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 vocabulary=vocabulary,
                                 stop_words='english',
                                 max_df=1.0,binary=True,
                                 min_df=1 )
    print vectorizer
    x_test=vectorizer.fit_transform(x_test)
    x_test=x_test.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    x_train=transformer.fit_transform(x_train)
    x_train=x_train.toarray()
    x_test=transformer.transform(x_test)
    x_test=x_test.toarray()

    return x_train, x_test, y_train, y_test


def get_features_by_ngram():
    global max_features
    x_train, x_test, y_train, y_test=load_all_files()

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                ngram_range=(3, 3),
                                strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1,
                                 token_pattern=r'\b\w+\b',
                                 binary=True)
    print vectorizer
    x_train=vectorizer.fit_transform(x_train)
    x_train=x_train.toarray()
    vocabulary=vectorizer.vocabulary_

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                ngram_range=(3, 3),
                                strip_accents='ascii',
                                 vocabulary=vocabulary,
                                 stop_words='english',
                                 max_df=1.0,binary=True,
                                 token_pattern=r'\b\w+\b',
                                 min_df=1 )
    print vectorizer
    x_test=vectorizer.fit_transform(x_test)
    x_test=x_test.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    x_train=transformer.fit_transform(x_train)
    x_train=x_train.toarray()
    x_test=transformer.transform(x_test)
    x_test=x_test.toarray()

    return x_train, x_test, y_train, y_test

def do_cnn_wordbag(trainX, testX, trainY, testY):
    global max_document_length
    print "CNN and tf"

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_document_length], name='input')
    network = tflearn.embedding(network, input_dim=1000000, output_dim=128)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=100,run_id="review")

def do_cnn_doc2vec_2d(trainX, testX, trainY, testY):
    print "CNN and doc2vec 2d"

    trainX = trainX.reshape([-1, max_features, max_document_length, 1])
    testX = testX.reshape([-1, max_features, max_document_length, 1])


    # Building convolutional network
    network = input_data(shape=[None, max_features, max_document_length, 1], name='input')
    network = conv_2d(network, 16, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
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
    model.fit({'input': trainX}, {'target': trainY}, n_epoch=20,
               validation_set=({'input': testX}, {'target': testY}),
               snapshot_step=100, show_metric=True, run_id='review')

def do_cnn_word2vec_2d(trainX, testX, trainY, testY):
    global max_features
    global max_document_length
    print "CNN and word2vec2d"
    y_test = testY
    #trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    #testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_document_length,max_features,1], name='input')

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
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True,run_id="sms")

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


def do_cnn_word2vec_2d_345(trainX, testX, trainY, testY):
    global max_features
    global max_document_length
    print "CNN and word2vec_2d_345"
    y_test = testY

    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_document_length,max_features,1], name='input')
    network = tflearn.embedding(network, input_dim=1, output_dim=128,validate_indices=False)
    branch1 = conv_2d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_2d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_2d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool_2d(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=100,run_id="sms")

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

def do_cnn_word2vec(trainX, testX, trainY, testY):
    global max_features
    print "CNN and word2vec"
    y_test = testY
    #trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    #testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_features], name='input')
    network = tflearn.embedding(network, input_dim=1000000, output_dim=128,validate_indices=False)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=100,run_id="sms")

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



def do_cnn_doc2vec(trainX, testX, trainY, testY):
    global max_features
    print "CNN and doc2vec"

    #trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    #testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_features], name='input')
    network = tflearn.embedding(network, input_dim=1000000, output_dim=128,validate_indices=False)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=100,run_id="review")

def do_rnn_wordbag(trainX, testX, trainY, testY):
    global max_document_length
    print "RNN and wordbag"
    y_test=testY
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, max_document_length])
    net = tflearn.embedding(net, input_dim=10240000, output_dim=128)
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

def do_rnn_word2vec(trainX, testX, trainY, testY):
    global max_features
    print "RNN and wordbag"

    trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, max_features])
    net = tflearn.embedding(net, input_dim=10240000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=10,run_id="sms",n_epoch=5)

def do_dnn_wordbag(x_train, x_test, y_train, y_test):
    print "MLP and wordbag"

    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    print  clf
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

def do_dnn_doc2vec(x_train, x_test, y_train, y_test):
    print "MLP and doc2vec"
    global max_features
    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    print  clf
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

def do_dnn_word2vec(x_train, x_test, y_train, y_test):
    print "MLP and word2vec"
    global max_features
    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    print  clf
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)

def do_dnn_wordbag(x_train, x_test, y_train, y_test):
    print "MLP and wordbag"
    global max_features
    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    print  clf
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)

def do_xgboost_word2vec(x_train, x_test, y_train, y_test):
    print "xgboost and word2vec"
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)

def do_xgboost_wordbag(x_train, x_test, y_train, y_test):
    print "xgboost and wordbag"
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)


def  get_features_by_tf():
    global  max_document_length
    x_train, x_test, y_train, y_test=load_all_files()

    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x_train=vp.fit_transform(x_train, unused_y=None)
    x_train=np.array(list(x_train))

    x_test=vp.transform(x_test)
    x_test=np.array(list(x_test))
    return x_train, x_test, y_train, y_test


def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text

def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        #labelized.append(LabeledSentence(v, [label]))
        #labelized.append(LabeledSentence(words=v,tags=label))
        labelized.append(SentimentDocument(v, [label]))
    return labelized

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.array(np.concatenate(vecs),dtype='float')


def buildWordVector(imdb_w2v,text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def  get_features_by_doc2vec():
    global  max_features
    x_train, x_test, y_train, y_test=load_all_files()

    x_train=cleanText(x_train)
    x_test=cleanText(x_test)

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')

    x=x_train+x_test
    cores=multiprocessing.cpu_count()
    #models = [
        # PV-DBOW
    #    Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=19, iter=10, workers=cores),
        # PV-DM w/average
    #    Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=19, iter=10, workers=cores),
    #]
    if os.path.exists(doc2ver_bin):
        print "Find cache file %s" % doc2ver_bin
        model=Doc2Vec.load(doc2ver_bin)
    else:
        model=Doc2Vec(dm=0, size=max_features, negative=5, hs=0, min_count=2, workers=cores,iter=60)


        #for model in models:
        #    model.build_vocab(x)
        model.build_vocab(x)

        #models[1].reset_from(models[0])

        #for model in models:
        #    model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        #models[0].train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(doc2ver_bin)

    #x_test=getVecs(models[0],x_test,max_features)
    #x_train=getVecs(models[0],x_train,max_features)
    x_test=getVecs(model,x_test,max_features)
    x_train=getVecs(model,x_train,max_features)

    return x_train, x_test, y_train, y_test

def  get_features_by_word2vec():
    global  max_features
    global word2ver_bin
    x_train, x_test, y_train, y_test=load_all_files()

    x_train=cleanText(x_train)
    x_test=cleanText(x_test)

    x=x_train+x_test
    cores=multiprocessing.cpu_count()

    if os.path.exists(word2ver_bin):
        print "Find cache file %s" % word2ver_bin
        model=gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model=gensim.models.Word2Vec(size=max_features, window=10, min_count=1, iter=60, workers=cores)

        model.build_vocab(x)

        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(word2ver_bin)


    x_train= np.concatenate([buildWordVector(model,z, max_features) for z in x_train])
    x_train = scale(x_train)
    x_test= np.concatenate([buildWordVector(model,z, max_features) for z in x_test])
    x_test = scale(x_test)

    return x_train, x_test, y_train, y_test

def  get_features_by_word2vec_cnn_1d():
    global  max_features
    global word2ver_bin
    x_train, x_test, y_train, y_test=load_all_files()

    x_train=cleanText(x_train)
    x_test=cleanText(x_test)

    x=x_train+x_test
    cores=multiprocessing.cpu_count()

    if os.path.exists(word2ver_bin):
        print "Find cache file %s" % word2ver_bin
        model=gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model=gensim.models.Word2Vec(size=max_features, window=10, min_count=1, iter=60, workers=cores)

        model.build_vocab(x)

        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(word2ver_bin)

    min_max_scaler = preprocessing.MinMaxScaler()

    x_train= np.concatenate([buildWordVector(model,z, max_features) for z in x_train])
    x_train = min_max_scaler.fit_transform(x_train)
    x_test= np.concatenate([buildWordVector(model,z, max_features) for z in x_test])
    x_test = min_max_scaler.transform(x_test)

    return x_train, x_test, y_train, y_test



def  get_features_by_word2vec_cnn_2d():
    global max_features
    global max_document_length
    global word2ver_bin

    x_train, x_test, y_train, y_test=load_all_files()

    x_train_vecs=[]
    x_test_vecs=[]

    x_train=cleanText(x_train)
    x_test=cleanText(x_test)

    x=x_train+x_test
    cores=multiprocessing.cpu_count()

    if os.path.exists(word2ver_bin):
        print "Find cache file %s" % word2ver_bin
        model=gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model=gensim.models.Word2Vec(size=max_features, window=10, min_count=1, iter=60, workers=cores)

        model.build_vocab(x)

        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(word2ver_bin)


    #x_train_vec=np.zeros((max_document_length,max_features))
    #x_test_vec=np.zeros((max_document_length, max_features))
    """
    x_train= np.concatenate([buildWordVector(model,z, max_features) for z in x_train])
    x_train = min_max_scaler.fit_transform(x_train)
    x_test= np.concatenate([buildWordVector(model,z, max_features) for z in x_test])
    x_test = min_max_scaler.transform(x_test)
    vec += imdb_w2v[word].reshape((1, size))
    """
    #x_train = np.concatenate([buildWordVector_2d(model, z, max_features) for z in x_train])
    x_all=np.zeros((1,max_features))
    for sms in x_train:
        sms=sms[:max_document_length]
        #print sms
        x_train_vec = np.zeros((max_document_length, max_features))
        for i,w in enumerate(sms):
            vec=model[w].reshape((1, max_features))
            x_train_vec[i-1]=vec.copy()
            #x_all=np.concatenate((x_all,vec))
        x_train_vecs.append(x_train_vec)
        #print x_train_vec.shape
    for sms in x_test:
        sms=sms[:max_document_length]
        #print sms
        x_test_vec = np.zeros((max_document_length, max_features))
        for i,w in enumerate(sms):
            vec=model[w].reshape((1, max_features))
            x_test_vec[i-1]=vec.copy()
            #x_all.append(vec)
        x_test_vecs.append(x_test_vec)

    #print x_train
    #print x_all
    min_max_scaler = preprocessing.MinMaxScaler()
    print "fix min_max_scaler"
    x_train_2d=np.concatenate([z for z in x_train_vecs])
    min_max_scaler.fit(x_train_2d)

    x_train=np.concatenate([min_max_scaler.transform(i) for i in x_train_vecs])
    x_test=np.concatenate([min_max_scaler.transform(i) for i in x_test_vecs])

    x_train=x_train.reshape([-1, max_document_length, max_features, 1])
    x_test = x_test.reshape([-1, max_document_length, max_features, 1])

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    print "Hello sms"
    x_train, x_test, y_train, y_test =get_features_by_tf()
    #print "get_features_by_ngram"
    #x_train, x_test, y_train, y_test=get_features_by_ngram()
    #print "get_features_by_word2vec"
    #x_train, x_test, y_train, y_test=get_features_by_word2vec()
    #x_train, x_test, y_train, y_test =get_features_by_word2vec_cnn_1d()
    #x_train, x_test, y_train, y_test = get_features_by_word2vec_cnn_2d()
    #x_train, x_test, y_train, y_test = get_features_by_word2vec_cnn_1d_log()
    #print x_train
    #print "get_features_by_wordbag"

    #x_train, x_test, y_train, y_test=get_features_by_wordbag_tfidf()
    #NB
    #do_nb_doc2vec(x_train, x_test, y_train, y_test)
    #do_nb_word2vec(x_train, x_test, y_train, y_test)
    #do_nb_wordbag(x_train, x_test, y_train, y_test)
    #CNN
    #do_cnn_word2vec(x_train, x_test, y_train, y_test)
    #do_cnn_doc2vec_2d(x_train, x_test, y_train, y_test)
    #do_cnn_word2vec_2d(x_train, x_test, y_train, y_test)
    #do_cnn_word2vec_2d_345(x_train, x_test, y_train, y_test)
    #DNN
    #do_dnn_doc2vec(x_train, x_test, y_train, y_test)
    #do_dnn_word2vec(x_train, x_test, y_train, y_test)
    #do_dnn_wordbag(x_train, x_test, y_train, y_test)
    #SVM
    #do_svm_doc2vec(x_train, x_test, y_train, y_test)
    #do_svm_word2vec(x_train, x_test, y_train, y_test)
    #rf
    #do_rf_doc2vec(x_train, x_test, y_train, y_test)
    #do_rf_word2vec(x_train, x_test, y_train, y_test)
    #do_xgboost_word2vec(x_train, x_test, y_train, y_test)
    #do_xgboost_wordbag(x_train, x_test, y_train, y_test)
    #RNN
    #do_rnn_word2vec(x_train, x_test, y_train, y_test)
    do_rnn_wordbag(x_train, x_test, y_train, y_test)
