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

max_features=15000

webshell_dir="../data/webshell/webshell/PHP/"
whitefile_dir="../data/webshell/normal/php/"
check_dir="../../../../../Downloads/php-exploit-scripts-master/"
white_count=0
black_count=0

def load_files_re(dir):
    files_list = []
    g = os.walk(dir)
    for path, d, filelist in g:
        #print d;
        for filename in filelist:
            #print os.path.join(path, filename)
            if filename.endswith('.php') or filename.endswith('.txt'):
                fulepath = os.path.join(path, filename)
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
    global white_count
    global black_count
    x=[]
    y=[]

    webshell_files_list = load_files_re(webshell_dir)
    y1=[1]*len(webshell_files_list)
    black_count=len(webshell_files_list)

    wp_files_list =load_files_re(whitefile_dir)
    y2=[0]*len(wp_files_list)

    white_count=len(wp_files_list)


    x=webshell_files_list+wp_files_list
    y=y1+y2

    CV = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",max_features=max_features,
                                       token_pattern = r'\b\w+\b',min_df=1, max_df=1.0)
    x=CV.fit_transform(x).toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    x_tfidf = transformer.fit_transform(x)
    x = x_tfidf.toarray()

    return x,y

def check_webshell(clf,dir):
    all=0
    all_php=0
    webshell=0

    webshell_files_list = load_files_re(webshell_dir)
    CV = CountVectorizer(ngram_range=(2, 2), decode_error="ignore", max_features=max_features,
                         token_pattern=r'\b\w+\b', min_df=1, max_df=1.0)
    x = CV.fit_transform(webshell_files_list).toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    transformer.fit_transform(x)


    g = os.walk(dir)
    for path, d, filelist in g:
        for filename in filelist:
            fulepath=os.path.join(path, filename)
            t = load_file(fulepath)
            t_list=[]
            t_list.append(t)
            x2 = CV.transform(t_list).toarray()
            x2 = transformer.transform(x2).toarray()
            y_pred = clf.predict(x2)
            all+=1
            if filename.endswith('.php'):
                all_php+=1
            if y_pred[0] == 1:
                print "%s is webshell" % fulepath
                webshell+=1

    print "Scan %d files(%d php files),%d files is webshell" %(all,all_php,webshell)


def do_check(x,y,clf):
    clf.fit(x, y)
    print "check_webshell"
    #check_webshell(clf,"../data/webshell/normal/php/")
    #/Users/maidou/Downloads/webshell-master/php
    check_webshell(clf,check_dir)



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
    print "load %d white %d black" % ( white_count,black_count )

    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)

    #print clf
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    do_metrics(y_test,y_pred)


    #do_check(x,y,clf)








