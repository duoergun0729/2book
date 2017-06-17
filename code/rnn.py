from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
import os
import pickle
from six.moves import urllib

import tflearn
from tflearn.data_utils import *

# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

def lstm(trainX, trainY,testX, testY):
    # Data preprocessing
    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, 100])
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=32,run_id="rnn-lstm")

def bi_lstm(trainX, trainY,testX, testY):
    trainX = pad_sequences(trainX, maxlen=200, value=0.)
    testX = pad_sequences(testX, maxlen=200, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data(shape=[None, 200])
    net = tflearn.embedding(net, input_dim=20000, output_dim=128)
    net = tflearn.bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
    model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=64,run_id="rnn-bilstm")

def shakespeare():


    path = "shakespeare_input.txt"
    #path = "shakespeare_input-100.txt"
    char_idx_file = 'char_idx.pickle'

    if not os.path.isfile(path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/shakespeare_input.txt", path)

    maxlen = 25

    char_idx = None
    if os.path.isfile(char_idx_file):
        print('Loading previous char_idx')
        char_idx = pickle.load(open(char_idx_file, 'rb'))

    X, Y, char_idx = \
        textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3,
                                             pre_defined_char_idx=char_idx)

    pickle.dump(char_idx, open(char_idx_file, 'wb'))

    g = tflearn.input_data([None, maxlen, len(char_idx)])
    g = tflearn.lstm(g, 512, return_seq=True)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.lstm(g, 512, return_seq=True)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.lstm(g, 512)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
    g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                           learning_rate=0.001)

    m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                                  seq_maxlen=maxlen,
                                  clip_gradients=5.0,
                                  checkpoint_path='model_shakespeare')

    for i in range(50):
        seed = random_sequence_from_textfile(path, maxlen)
        m.fit(X, Y, validation_set=0.1, batch_size=128,
              n_epoch=1, run_id='shakespeare')
        print("-- TESTING...")
        print("-- Test with temperature of 1.0 --")
        print(m.generate(600, temperature=1.0, seq_seed=seed))
        #print(m.generate(10, temperature=1.0, seq_seed=seed))
        print("-- Test with temperature of 0.5 --")
        print(m.generate(600, temperature=0.5, seq_seed=seed))

#lstm(trainX, trainY,testX, testY)
#bi_lstm(trainX, trainY,testX, testY)
shakespeare()
