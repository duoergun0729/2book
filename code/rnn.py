from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

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

#lstm(trainX, trainY,testX, testY)
bi_lstm(trainX, trainY,testX, testY)

