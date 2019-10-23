import sys
import pandas as pd
import sklearn.metrics as metrics

N_words = {'fuck', 'ass', 'bitch', 'bastard', 'gay', 'cock', 'dick', 'suck', 'cunt', 'bullshit', 'jerk', 'idiot',
           'dumb', 'shit', 'retard', 'rape', 'dumbass', 'nigger', 'pussy', 'whore'}


def main(inFile):
    print("Reading File:", inFile)
    train = pd.read_csv(inFile)
    train = train.dropna()

    predictions = []
    threshold = 0
    labels = train.iloc[:, 1].tolist()
    for i, row in enumerate(train.itertuples(index=False)):
        text = row[0].split()
        count_toxic_word = len([word for word in text if word in N_words])
        predictions.append(1 if count_toxic_word > threshold else 0)
        print("Processed: {} / {}".format(i + 1, train.shape[0]))
    print("Accuracy: {}".format(metrics.accuracy_score(labels, predictions)))
    print("Precision: {}".format(metrics.precision_score(labels, predictions)))
    print("Recall: {}".format(metrics.recall_score(labels, predictions)))
    print("F1: {}".format(metrics.f1_score(labels, predictions)))


if __name__ == '__main__':
    inFile = sys.argv[1]
    main(inFile)