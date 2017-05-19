import os
from sklearn.feature_extraction.text import CountVectorizer
import sys
import numpy as np
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier

def load_files_re(dir):
    files_list = []
    g = os.walk(dir)
    for path, d, filelist in g:
        #print d;
        for filename in filelist:
            #print os.path.join(path, filename)
            fulepath=os.path.join(path, filename)
            print "Load %s" % fulepath
            t = load_file(fulepath)
            files_list.append(t)
    return files_list


def load_file(file_path):
    t=""
    with open(file_path) as f:
        for line in f:
            line=line.strip('\n')
            t+=line
    return t


def load_files(path):
    files_list=[]
    for r, d, files in os.walk(path):
        for file in files:
            if file.endswith('.php'):
                file_path=path+file
                print "Load %s" % file_path
                t=load_file(file_path)
                files_list.append(t)
    return  files_list

def get_feature_by_bag_tfidf():
    x=[]
    y=[]
    #bigram_vectorizer = CountVectorizer(ngram_range=(2, 2),token_pattern = r'\b\w+\b', min_df = 1)
    webshell_bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                                        token_pattern = r'\b\w+\b',min_df=1)
    #webshell_files_list=load_files("../data/PHP-WEBSHELL/xiaoma/")
    webshell_files_list = load_files_re("../data/webshell/webshell/PHP/")
    x1=webshell_bigram_vectorizer.fit_transform(webshell_files_list).toarray()
    y1=[1]*len(x1)
    vocabulary=webshell_bigram_vectorizer.vocabulary_

    wp_bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                                        token_pattern = r'\b\w+\b',min_df=1,vocabulary=vocabulary)
    #wp_files_list=load_files("../data/wordpress/")
    wp_files_list =load_files_re("../data/webshell/normal/php/")
    x2=wp_bigram_vectorizer.fit_transform(wp_files_list).toarray()
    y2=[0]*len(x2)

    x=np.concatenate((x1,x2))
    y=np.concatenate((y1, y2))

    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(x)
    x = tfidf.toarray()
    return x,y

def check_webshell(clf,dir):
    all=0
    webshell=0
    webshell_bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                                        token_pattern = r'\b\w+\b',min_df=1)
    webshell_files_list = load_files_re("../data/webshell/webshell/PHP/")
    x1 = webshell_bigram_vectorizer.fit_transform(webshell_files_list).toarray()

    vocabulary=webshell_bigram_vectorizer.vocabulary_

    check_bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                                        token_pattern = r'\b\w+\b',min_df=1,vocabulary=vocabulary)

    #check_files_list =load_files_re(dir)
    #x2=check_bigram_vectorizer.fit_transform(check_files_list).toarray()


    #x=np.concatenate((x1,x2))

    transformer = TfidfTransformer(smooth_idf=False)
    transformer.fit(x1)
    #x2 = tfidf.toarray()
    #y_pred = clf.predict(x2)

    g = os.walk(dir)
    for path, d, filelist in g:
        for filename in filelist:
            fulepath=os.path.join(path, filename)
            #print "Check %s" % fulepath
            t = load_file(fulepath)
            t_list=[]
            t_list.append(t)
            x2 = check_bigram_vectorizer.fit_transform(t_list).toarray()
            x2 = transformer.transform(x2).toarray()
            y_pred = clf.predict(x2)
            #print y_pred
            all+=1
            if y_pred[0] == 1:
                print "%s is webshell" % fulepath
                webshell+=1

    print "Scan %d files,%d files is webshell" %(all,webshell)


def do_check(x,y,clf):
    clf.fit(x, y)
    print "check_webshell"
    #check_webshell(clf,"../data/webshell/normal/php/")
    #/Users/maidou/Downloads/webshell-master/php
    check_webshell(clf,"/Users/maidou/Downloads/webshell-master/php/")



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

if __name__ == '__main__':

    x,y=get_feature_by_bag_tfidf()

    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    #clf.fit(x_train, y_train)
    #y_pred = clf.predict(x_test)



    do_check(x,y,clf)








