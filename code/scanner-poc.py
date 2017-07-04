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
#xss_data_file="../data/aiscanner/xss-script.txt"



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


    #pickle.dump(char_idx, open(char_idx_file, 'wb'))

    g = tflearn.input_data([None, maxlen, len(char_idx)])
    g = tflearn.lstm(g, 32, return_seq=True)
    g = tflearn.dropout(g, 0.1)
    g = tflearn.lstm(g, 32, return_seq=True)
    g = tflearn.dropout(g, 0.1)
    g = tflearn.lstm(g, 32)
    g = tflearn.dropout(g, 0.1)
    g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
    g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                           learning_rate=0.001)

    m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                                  seq_maxlen=maxlen,
                                  clip_gradients=5.0,
                                  checkpoint_path='chkpoint/model_scanner_poc')

    print "random_sequence_from_textfile"
    #seed = random_sequence_from_textfile(xss_data_file, maxlen)
    seed='"/><script>'
    m.fit(X, Y, validation_set=0.1, batch_size=128,
              n_epoch=2, run_id='scanner-poc')
    print("-- TESTING...")

    print("-- Test with temperature of 0.1 --")
    print(m.generate(32, temperature=0.1, seq_seed=seed))
    print("-- Test with temperature of 0.5 --")
    print(m.generate(32, temperature=0.5, seq_seed=seed))
    print("-- Test with temperature of 1.0 --")
    print(m.generate(32, temperature=1.0, seq_seed=seed))


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
    newsgroups_test = fetch_20newsgroups(subset='test')

    newsgroups=[]
    newsgroups.append(newsgroups_train.data)
    newsgroups.append(newsgroups_test.data)
    #newsgroups_train = fetch_20newsgroups()
    #print len(newsgroups_train.data)
    print newsgroups_train.data
    sentences=[re.findall("[a-z\-]+",s.lower()) for s in newsgroups_train.data]
    #sentences = [s.lower().split() for s in newsgroups_train.data]
    #print sentences

    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=1, workers=4,iter=20)

    #print len(sentences)

    for key in keywords:
        print "[%s] most_similar:" % key
        results=model.most_similar(positive=[key], topn=10)
        for i in results:
            print i

def get_login_pages_imdb(keywords):

    import gensim
    import re
    from tflearn.datasets import imdb

    train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                    valid_portion=0.1)

    trainX, trainY = train
    sentences=trainX
    print len(sentences)
    print sentences

    model = gensim.models.Word2Vec(sentences, size=200, window=3, min_count=1, workers=4,iter=50)



    for key in keywords:
        print "[%s] most_similar:" % key
        results=model.most_similar(positive=[key], topn=10)
        for i in results:
            print i

if __name__ == "__main__":
    print "Hello ai scanner poc"
    #generator_xss()
    get_login_pages(["user","password","email","name"])
    #get_login_pages_imdb(["user", "password", "email", "name"])