import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
import sys
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(500)

def train(trainFile):
    train = pd.read_csv(trainFile)
    tfidf = TfidfVectorizer(max_features=50000)
    train.dropna(inplace=True)
    train = train.sample(frac=1)
    X_train = tfidf.fit_transform(train['comment_text'])
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X_train, train['class'])    
    return tfidf, clf

def test(tfidf, clf, testFile):
    test = pd.read_csv(testFile)
    test.dropna(inplace=True)
    X_test = tfidf.transform(test['comment_text'])
    y_test = np.array(test['class'])
    pred = clf.predict(X_test)
    cm = confusion_matrix(y_test,pred)
    p = cm[1][1]/(cm[1][1]+cm[0][1])
    r = cm[1][1]/(cm[1][1]+cm[1][0])
    a = (cm[1][1]+cm[0][0])/(cm[1][1]+cm[1][0]+cm[0][0]+cm[0][1])
    print("Precision: ", p)
    print("Recall: ", r)
    print("F1: ", 2*p*r/(p+r))
    print("Accuracy:", a)

def main(trainFilePath, testFilePath):
    tfidf, clf = train(trainFilePath)
    test(tfidf, clf, testFilePath)    

if __name__ == "__main__":
    trainFilePath = sys.argv[1]
    testFilePath = sys.argv[2]
    main(trainFilePath, testFilePath)