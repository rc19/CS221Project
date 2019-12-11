import matplotlib.pyplot as plt
import numpy as np

x_axis = [0.05, 0.1, 0.25, 0.5, 0.7, 1]
bi_lstm = [0.755, 0.787, 0.782, 0.771, 0.776, 0.815]
svm = [0.724, 0.744, 0.746, 0.745, 0.746, 0.813]
lr = [0.677, 0.707, 0.726, 0.736, 0.738, 0.81]

texts = ['Baseline', 'BT(ES)', 'BT(All)', 'EDA', 'BT(All)+EDA', 'Oracle']

plt.figure()
plt.plot(x_axis, bi_lstm, marker='o', label='LSTM')
plt.plot(x_axis, svm, marker='s', label='SVM', color='orange')
plt.plot(x_axis, lr, marker='^', label='Logistic Regression', color='red')
plt.xticks(np.arange(0, 1.1, step=0.1))

plt.legend(loc=4)
plt.xlabel('Fraction of Dataset Size')
plt.ylabel('F1 Score')

for i, text in enumerate(texts):
    plt.annotate(text, (x_axis[i], bi_lstm[i]))

for i, text in enumerate(texts[:-1]):
    plt.annotate(text, (x_axis[i], svm[i]))
    plt.annotate(text, (x_axis[i], lr[i]))

plt.savefig('performance.png')