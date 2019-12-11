import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, LSTM, Dense, Embedding, GlobalMaxPool1D
from sklearn import metrics
import pandas as pd
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

embed_size = 100
max_features = 50000
maxlen = 100


np.random.seed(2019)


def read_glove(embedding_file):
    features = {}
    with open(embedding_file, encoding="utf-8") as e:
        for i, line in enumerate(e):
            text = line.split()
            features[text[0]] = np.array(text[1:], float)
    return features


def get_word_embedding(train, test, word_param):
    comment_train = train['comment_text'].values
    comment_test = test['comment_text'].values
    tokenizer = Tokenizer(max_features)
    tokenizer.fit_on_texts(list(comment_train))
    comment_tokenized_train = tokenizer.texts_to_sequences(comment_train)
    comment_tokenized_test = tokenizer.texts_to_sequences(comment_test)
    train = pad_sequences(comment_tokenized_train, maxlen=maxlen)
    test = pad_sequences(comment_tokenized_test, maxlen=maxlen)
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))

    embedding = np.zeros((nb_words, embed_size))

    for word, index in word_index.items():
        if word in word_param and index < nb_words:
            embedding[index] = word_param[word]

    return train, test, embedding, word_index, tokenizer


def build_lstm(word_embedding):
    model = keras.Sequential()
    model.add(Embedding(word_embedding.shape[0], embed_size, weights=[word_embedding], trainable=False))
    model.add(Bidirectional(LSTM(embed_size, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(400, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.binary_accuracy])
    return model


def main():
    train_baseline = 'data/train_baseline.csv'
    train_eda = 'data/eda_train_baseline.csv'
    train_oracle = 'data/train_oracle.csv'
    test_data = 'data/test.csv'
    glove_file = 'glove.6B/glove.6B.100d.txt'
    train_back_fr = 'data/train_baseline_fr.csv'
    train_back_hi = 'data/train_baseline_hi.csv'
    train_back_de = 'data/train_baseline_de.csv'
    train_back_es = 'data/train_baseline_es.csv'

    train = pd.concat([pd.read_csv(train_eda), pd.read_csv(train_back_fr), pd.read_csv(train_back_hi), pd.read_csv(train_back_de),
                       pd.read_csv(train_back_es)])
    test = pd.read_csv(test_data)
    train.dropna()
    test.dropna()
    glove_param = read_glove(glove_file)

    train_labels = train['class'].values
    test_labels = test['class'].values
    train_words, test_words, word_embedding, word_index, tokenizer = get_word_embedding(train, test, glove_param)
    model = build_lstm(word_embedding)

    model.fit(train_words, train_labels, batch_size=250, epochs=1)

    test_predict = model.predict_classes(test_words)[:, 0]

    print("Accuracy: {}".format(metrics.accuracy_score(test_labels, test_predict)))
    print("Precision: {}".format(metrics.precision_score(test_labels, test_predict)))
    print("Recall: {}".format(metrics.recall_score(test_labels, test_predict)))
    print("F1: {}".format(metrics.f1_score(test_labels, test_predict)))

    # Analyze
    def new_predict(texts):
        _seq = tokenizer.texts_to_sequences(texts)
        _text_data = pad_sequences(_seq, maxlen=maxlen)
        prob = model.predict(_text_data)
        prob = np.concatenate([1 - prob, prob], axis=-1)
        return prob

    class_names = ['non-toxic', 'toxic']
    explainer = LimeTextExplainer(class_names=class_names)

    idx = 10
    exp = explainer.explain_instance(test['comment_text'][idx], new_predict, num_features=30)
    exp_l = exp.as_list()

    names = [l[0] for l in exp_l if l[1] > 0]
    scores = [l[1] for l in exp_l if l[1] > 0]
    plt.figure(figsize=(15, 6))
    plt.bar(names, scores)
    plt.title("Local explanation for class toxic")
    plt.savefig('eda_back_translation_all.png')


if __name__ == '__main__':
    main()
