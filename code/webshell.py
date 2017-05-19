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



if __name__ == '__main__':

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

    #clf = GaussianNB()
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)

    #print  cross_validation.cross_val_score(clf, x, y, n_jobs=-1,cv=10)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

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






