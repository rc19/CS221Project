import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, LSTM, Dense, Embedding, GlobalMaxPool1D
from sklearn import metrics
import pandas as pd

embed_size = 100
max_features = 50000
maxlen = 100


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

    embs = np.stack(word_param.values())
    mean, std = embs.mean(), embs.std()
    embedding = np.random.normal(mean, std, size=(nb_words, embed_size))

    for word, index in word_index.items():
        if word in word_param and index < nb_words:
            embedding[index] = word_param[word]

    return train, test, embedding, nb_words


def build_lstm(input_dim, word_embedding):
    model = keras.Sequential()
    model.add(Embedding(input_dim, embed_size, weights=[word_embedding], trainable=False))
    model.add(Bidirectional(LSTM(embed_size, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(400, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.binary_accuracy])
    return model


def main():
    train_data = 'data/train_oracle.csv'
    test_data = 'data/test.csv'
    glove_file = 'glove.6B/glove.6B.100d.txt'

    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)
    train.dropna()
    test.dropna()
    glove_param = read_glove(glove_file)

    train_labels = train['class'].values
    test_labels = test['class'].values
    train_word_index, test_word_index, word_embedding, nb_words = get_word_embedding(train, test, glove_param)

    model = build_lstm(nb_words, word_embedding)

    model.fit(train_word_index, train_labels, batch_size=64, epochs=5)

    test_predict = model.predict_classes(test_word_index)[:, 0]

    print("Accuracy: {}".format(metrics.accuracy_score(test_labels, test_predict)))
    print("Precision: {}".format(metrics.precision_score(test_labels, test_predict)))
    print("Recall: {}".format(metrics.recall_score(test_labels, test_predict)))
    print("F1: {}".format(metrics.f1_score(test_labels, test_predict)))


if __name__ == '__main__':
    main()
