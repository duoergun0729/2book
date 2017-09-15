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
from hmmlearn import hmm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def get_feature():
    df = pd.read_csv("../data/fraud/creditcard.csv")
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    y=df['Class']
    #print y
    features = df.drop(['Class'], axis=1).columns
    x=df[features]
    #print x

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    return x_train, x_test, y_train, y_test


def do_xgboost(x_train, x_test, y_train, y_test):
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    do_metrics(y_test, y_pred)

def do_mlp(x_train, x_test, y_train, y_test):
    #mlp
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    do_metrics(y_test,y_pred)

def do_nb(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    do_metrics(y_test,y_pred)

def do_metrics(y_test,y_pred):
    print "metrics.accuracy_score:"
    print metrics.accuracy_score(y_test, y_pred)
    print "metrics.confusion_matrix:"
    print metrics.confusion_matrix(y_test, y_pred)
    print "metrics.precision_score:"
    print metrics.precision_score(y_test, y_pred)
    print "metrics.recall_score:"
    print metrics.recall_score(y_test, y_pred)
    print "metrics.f1_score:"
    print metrics.f1_score(y_test,y_pred)


def run_1():
    x_train, x_test, y_train, y_test=get_feature()
    do_xgboost(x_train, x_test, y_train, y_test)
    do_mlp(x_train, x_test, y_train, y_test)
    do_nb(x_train, x_test, y_train, y_test)


def run_2():
    x_train, x_test, y_train, y_test=get_feature_undersampling()
    print "XGBoost"
    do_xgboost(x_train, x_test, y_train, y_test)
    print "mlp"
    do_mlp(x_train, x_test, y_train, y_test)
    print "nb"
    do_nb(x_train, x_test, y_train, y_test)

def run_3():
    x_train, x_test, y_train, y_test=get_feature_upsampling()
    print "XGBoost"
    do_xgboost(x_train, x_test, y_train, y_test)
    print "mlp"
    do_mlp(x_train, x_test, y_train, y_test)
    print "nb"
    do_nb(x_train, x_test, y_train, y_test)

def get_feature_undersampling():
    df = pd.read_csv("../data/fraud/creditcard.csv")
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    number_fraud=len(df[df.Class==1])
    #print number_fraud
    fraud_index=np.array(df[df.Class==1].index)
    #print fraud_index

    normal_index=df[df.Class==0].index
    random_choice_index=np.random.choice(normal_index,size=number_fraud,replace=False)

    x_index=np.concatenate([fraud_index,random_choice_index])
    df = df.drop(['Class'], axis=1)
    x=df.iloc[x_index,:]
    #print x
    y=[1]*number_fraud+[0]*number_fraud


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    return x_train, x_test, y_train, y_test

def get_feature_undersampling_2():
    df = pd.read_csv("../data/fraud/creditcard.csv")
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    y = df['Class']
    features = df.drop(['Class'], axis=1).columns
    x = df[features]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    print "raw data"
    print pd.value_counts(y_train)



    number_fraud=len(y_train[y_train==1])
    print number_fraud
    fraud_index=np.array(y_train[y_train==1].index)
    print fraud_index

    normal_index=y_train[y_train==0].index
    random_choice_index=np.random.choice(normal_index,size=number_fraud,replace=False)

    x_index=np.concatenate([fraud_index,random_choice_index])
    print x_index
    #df = df.drop(['Class'], axis=1)
    x_train_1=x.iloc[x_index,:]

    #print x
    y_train_1=[1]*number_fraud+[0]*number_fraud


    print "Undersampling data"
    print pd.value_counts(y_train_1)

    return x_train_1, x_test, y_train_1, y_test

def get_feature_upsampling():
    df = pd.read_csv("../data/fraud/creditcard.csv")
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    y = df['Class']
    features = df.drop(['Class'], axis=1).columns
    x = df[features]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    print "raw data"
    print pd.value_counts(y_train)

    os = SMOTE(random_state=0)
    x_train_1,y_train_1=os.fit_sample(x_train,y_train)
    print "Smote data"
    print pd.value_counts(y_train_1)


    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    #特征提取使用标准化
    #run_1()
    #特征提取使用标准化&降采样
    run_2()
    #特征提取使用标准化&过采样
    #run_3()


#print y_train

