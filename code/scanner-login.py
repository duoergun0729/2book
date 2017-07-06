# -*- coding:utf-8 -*-
import os
import pickle
from six.moves import urllib

import tflearn
from tflearn.data_utils import *

char_idx_file = 'char_idx_xss.pkl'
maxlen = 25
char_idx = None
xss_data_file="../data/aiscanner/xss.txt"


def generator_xss():
    global char_idx
    global xss_data_file
    global maxlen


    if os.path.isfile(char_idx_file):
        print('Loading previous xxs_char_idx')
        char_idx = pickle.load(open(char_idx_file, 'rb'))

    X, Y, char_idx = \
        textfile_to_semi_redundant_sequences(xss_data_file, seq_maxlen=maxlen, redun_step=3,
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
                                  checkpoint_path='model_scanner_poc')

    for i in range(50):
        seed = random_sequence_from_textfile(xss_data_file, maxlen)
        m.fit(X, Y, validation_set=0.1, batch_size=128,
              n_epoch=1, run_id='scanner-poc')
        print("-- TESTING...")
        print("-- Test with temperature of 1.0 --")
        print(m.generate(600, temperature=1.0, seq_seed=seed))
        print("-- Test with temperature of 0.5 --")
        print(m.generate(600, temperature=0.5, seq_seed=seed))

def get_login_pages(keywords):
    from sklearn.datasets import fetch_20newsgroups
    import gensim
    import re
    """
    newsgroups_train = fetch_20newsgroups(subset='train')
    for  news in newsgroups_train.target_names:
        print news

    alt.atheism
    comp.graphics
    comp.os.ms-windows.misc
    comp.sys.ibm.pc.hardware
    comp.sys.mac.hardware
    comp.windows.x
    misc.forsale
    rec.autos
    rec.motorcycles
    rec.sport.baseball
    rec.sport.hockey
    sci.crypt
    sci.electronics
    sci.med
    sci.space
    soc.religion.christian
    talk.politics.guns
    talk.politics.mideast
    talk.politics.misc
    talk.religion.misc
    """
    #cats = ['sci.crypt']
    #newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
    newsgroups_train = fetch_20newsgroups(subset='train')
    sentences=[re.findall("[a-z]+",s.lower()) for s in newsgroups_train.data]
    print sentences

    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=1, workers=4)
    results=model.most_similar(positive=[keywords], topn=10)
    for i in results:
        print i

if __name__ == "__main__":
    print "Hello ai scanner poc"
    #generator_xss()
    get_login_pages("password")