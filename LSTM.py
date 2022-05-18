#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

import tensorflow as tf

import sklearn.metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from numpy import mean
from numpy import std
from numpy import dstack
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix




dataframe = pd.read_csv('./KU_HAR.csv')
dataframe.head()


dataframe.shape




train, validate, test = np.split(df.sample(frac=1), [int(.78*len(df)), int(.8*len(df))])



train.shape


validate.shape


test.shape




scaler = StandardScaler().fit(train)
X_train = scaler.transform(train)
X_test = scaler.transform(test)
X_validate = scaler.transform(validate)



y_train = X_train[1800]
X_train = X_train.iloc[:,:1800]

y_test = X_test.shape[-1]
X_test = X_test.iloc[:,:-1]

y_valid = X_validate.shape[-1]
X_valid = X_validate.iloc[:,:-1]



# train the model
model = Sequential()
model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])




# fit model
verbose, epochs, batch_size = 0, 20, 32
n_timesteps, n_features, n_outputs = X_train.shape[0], X_train.shape[1], y_train.shape[0]
model = Sequential()
model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model_lstm = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
              validation_data=(X_valid, y_valid), shuffle=True)
y_pred = model_lstm.predict(X_test, verbose=0)
ypred_classes = model.predict_classes(X_test, verbose=0)


# repeat experiment
scores = list()
for r in range(repeats):
    score = evaluate_model(X_train, y_train, X_test, y_test)
    score = score * 100.0
    print('>#%d: %.3f' % (r+1, score))
    scores.append(score)


# print results
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# experiment
def run_experiment(repeats=10):
    scores = list()
    for r in range(repeats):
        score = evaluate_model(X_train, y_train, X_test, y_test)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    summarize_results(scores)



print(summarize_results(scores))



# evaluation
accuracy = accuracy_score(y_test, ypred_classes)
print('Accuracy: %f' % accuracy)

precision = precision_score(y_test, ypred_classes)
print('Precision: %f' % precision)

recall = recall_score(y_test, ypred_classes)
print('Recall: %f' % recall)

f1 = f1_score(y_test, ypred_classes)
print('F1 score: %f' % f1)



cm = confusion_matrix(y_test, y_pred, labels=model_lstm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_lstm.classes_)


fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax);


pyplot.plot(model_lstm.model_lstm['loss'])
pyplot.plot(model_lstm.model_lstm['validation_loss'])
pyplot.title('Model Train and Validation')
pyplot.ylabel('Validation Loss')
pyplot.xlabel('Epochs')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

