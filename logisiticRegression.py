import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import sklearn.metrics as metrics

def extract_features(field,training_data,testing_data):
    # get the features - word level tf-idf
    #TF-IDF score represents the relative importance  of a term in the document and the entire corpus.
    # TF-IDF score is composed by two terms: the first computes the normalized Term Frequency (TF),
    # the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents
    # in the corpus divided by the number of documents where the specific term appears.
    print("Getting the features..")
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)
    tfidf_vectorizer.fit_transform(training_data[field].values)
    train_feature_set = tfidf_vectorizer.transform(training_data[field].values)
    test_feature_set = tfidf_vectorizer.transform(testing_data[field].values)
    return train_feature_set, test_feature_set, tfidf_vectorizer

def main(trainFile,testFile):
    # get a train and test split
    print("Reading train File:", trainFile)
    training_data = pd.read_csv(trainFile)
    training_data = training_data.dropna()
    print("Reading test File:", testFile)
    testing_data = pd.read_csv(testFile)
    testing_data = testing_data.dropna()

    # get the labels ( Y - label)
    label = 'class'
    Y_train = training_data[label].values
    Y_test = testing_data[label].values

    # get the features (X - feature)
    # extract the different types of features based on the tf-id weighting scheme
    field = 'comment_text'
    X_train, X_test, feature_transformer = extract_features(field, training_data, testing_data)

    # train your logistic regression model
    print("Training a logisitic regression model...")
    model = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
    print("Here!")
    model.fit(X_train, Y_train)
    print("Here!")
    # return list of predictions
    predictions = model.predict(X_test)
    print("predictions",predictions)
    # compute the accuracy
    print("Accuracy: {}".format(metrics.accuracy_score(Y_test, predictions)))
    print("Precision: {}".format(metrics.precision_score(Y_test, predictions)))
    print("Recall: {}".format(metrics.recall_score(Y_test, predictions)))
    print("F1: {}".format(metrics.f1_score(Y_test, predictions)))


if __name__ == '__main__':
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    main(trainFile,testFile)