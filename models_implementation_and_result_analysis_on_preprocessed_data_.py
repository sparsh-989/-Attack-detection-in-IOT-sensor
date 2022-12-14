# -*- coding: utf-8 -*-
"""Models Implementation and Result Analysis on Preprocessed Data .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f-RIegN9veMO8czs32C7dUt9dt6YniEJ

# This Notebook starts from Preprocessed Dataframework. "df_spark.csv" is the dataframe
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

df_spark = pd.read_csv('/content/drive/My Drive/Code/df_spark.csv')

df_spark.head()

df_spark = df_spark.drop(columns="Unnamed: 0")

df_spark.head()

"""# In the following code X contains features and y contains label"""

y = df_spark.iloc[:,0].values
X = df_spark.iloc[:,1:].values

"""# The whole dataset is split into 80:20 ratio. X_train contains 80% of the features, X_test contains 20% of the features and y_train contains 80% corresponding label of X_train and y_test contains 20% corresponding label of X_test"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state = 1)

print(X_train.shape)
print(X_train[0])

print(X_test.shape)
print(X_test)

print(y_train.shape)
print(y_train)

print(y_test.shape)
print(y_test)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

"""# 5-Fold Cross validation Estimation for KNN"""

pipe_knn = Pipeline([('scl', StandardScaler()),('clf', KNeighborsClassifier(n_neighbors= 1))])

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_knn,X=X, y=y, train_sizes=np.linspace(0.2,1.0,5), cv=5, n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

for i in train_sizes:
    print(i)

for i in train_mean:
    print(i)

for i in test_mean:
    print(i)

"""# Evaluation Metrics Calculations for KNN"""

pipe_knn = pipe_knn.fit(X_train, y_train)

y_pred_train = pipe_knn.predict(X_train)

y_pred_test = pipe_knn.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_train, y_pred_train)

accuracy_score(y_test, y_pred_test)

from sklearn.metrics import classification_report

target_names = ['Normal', 'DoSattack', 'scan', 'malitiousControl', 'malitiousOperation', 'spying', 'dataProbing', 'wrongSetUp']

print(classification_report(y_train, y_pred_train, target_names=target_names))

print(classification_report(y_test, y_pred_test, target_names=target_names))

from sklearn.metrics import confusion_matrix
import itertools

cnf_matrix = confusion_matrix(y_test, y_pred_test)

for i in cnf_matrix:
    for j in i:
        print(j, end=' ')
    print()

"""# 5-Fold Cross validation Estimation for Gaussian Naive Bayes"""

from sklearn.naive_bayes import GaussianNB
pipe_gnb = Pipeline([('scl', StandardScaler()),('clf', GaussianNB(priors=None, var_smoothing=1e-09))])

#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', LogisticRegression(penalty='l2', random_state=0))])
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_gnb,X=X, y=y, train_sizes=np.linspace(0.2,1.0,5), cv=5, n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

for i in train_sizes:
    print(i)

for i in train_mean:
    print(i)

for i in test_mean:
    print(i)

"""# Evaluation Metrics Calculations for Gaussian Naive Bayes"""

pipe_gnb = pipe_gnb.fit(X_train, y_train)

y_pred_train = pipe_gnb.predict(X_train)

y_pred_test = pipe_gnb.predict(X_test)

accuracy_score(y_train, y_pred_train)

accuracy_score(y_test, y_pred_test)

print(classification_report(y_train, y_pred_train, target_names=target_names))

print(classification_report(y_test, y_pred_test, target_names=target_names))

cnf_matrix = confusion_matrix(y_test, y_pred_test)

for i in cnf_matrix:
    for j in i:
        print(j,end=' ')
    print()

"""# 5-Fold Cross validation Estimation for Logistic Regression"""

pipe_lr = Pipeline([('scl', StandardScaler()),('clf', LogisticRegression(penalty='l2', random_state=0))])
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,X=X, y=y, train_sizes=np.linspace(0.2,1.0,5), cv=5, n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

for i in train_sizes:
    print(i)

for i in train_mean:
    print(i)

for i in test_mean:
    print(i)



"""# 5-Fold Cross validation Estimation for SVM"""

from sklearn.svm import LinearSVC

pipe_svc = Pipeline([('scl', StandardScaler()),('clf', LinearSVC())])
train_sizes_svc, train_scores_svc, test_scores_svc = learning_curve(estimator=pipe_svc,X=X, y=y, train_sizes=np.linspace(0.2,1.0,5), cv=5, n_jobs=-1)
train_mean_svc = np.mean(train_scores_svc, axis=1)
train_std_svc = np.std(train_scores_svc, axis=1)
test_mean_svc = np.mean(test_scores_svc, axis=1)
test_std_svc = np.std(test_scores_svc, axis=1)

for i in train_mean_svc:
    print(i)

for i in test_mean_svc:
    print(i)

"""# 5-Fold Cross validation Estimation for Decision Tree"""

from sklearn import tree

pipe_tree = Pipeline([('scl', StandardScaler()),('clf', tree.DecisionTreeClassifier())])
train_sizes_tree, train_scores_tree, test_scores_tree = learning_curve(estimator=pipe_tree,X=X, y=y, train_sizes=np.linspace(0.2,1.0,5), cv=5, n_jobs=-1)
train_mean_tree = np.mean(train_scores_tree, axis=1)
train_std_tree = np.std(train_scores_tree, axis=1)
test_mean_tree = np.mean(test_scores_tree, axis=1)
test_std_tree = np.std(test_scores_tree, axis=1)

for i in train_mean_tree:
    print(i)

for i in test_mean_tree:
    print(i)

"""# 5-Fold Cross validation Estimation for Random Forest"""

from sklearn.ensemble import RandomForestClassifier

pipe_rnd = Pipeline([('scl', StandardScaler()),('clf', RandomForestClassifier(n_estimators=10))])
train_sizes_rnd, train_scores_rnd, test_scores_rnd = learning_curve(estimator=pipe_rnd,X=X, y=y, train_sizes=np.linspace(0.2,1.0,5), cv=5, n_jobs=-1)
train_mean_rnd = np.mean(train_scores_rnd, axis=1)
train_std_rnd = np.std(train_scores_rnd, axis=1)
test_mean_rnd = np.mean(test_scores_rnd, axis=1)
test_std_rnd = np.std(test_scores_rnd, axis=1)

for i in train_mean_rnd:
    print(i)

for i in test_mean_rnd:
    print(i)

"""# 5-Fold Cross validation Estimation for ANN"""

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1)

pipe_mlp = Pipeline([('scl', StandardScaler()),('clf', mlp)])
train_sizes_mlp, train_scores_mlp, test_scores_mlp = learning_curve(estimator=pipe_mlp,X=X, y=y, train_sizes=np.linspace(0.2,1.0,5), cv=5, n_jobs=-1)
train_mean_mlp = np.mean(train_scores_mlp, axis=1)
train_std_mlp = np.std(train_scores_mlp, axis=1)
test_mean_mlp = np.mean(test_scores_mlp, axis=1)
test_std_mlp = np.std(test_scores_mlp, axis=1)

for i in train_mean_mlp:
    print(i)

for i in test_mean_mlp:
    print(i)

"""# Mean values of Training and Testing accuracies and Standard Deviation of Training and Testing accuracies are given below"""

np.mean(train_mean) , np.mean(train_mean_svc), np.mean(train_mean_tree), np.mean(train_mean_rnd), np.mean(train_mean_mlp)

np.mean(train_std) , np.mean(train_std_svc), np.mean(train_std_tree), np.mean(train_std_rnd), np.mean(train_std_mlp)

np.mean(test_mean) , np.mean(test_mean_svc), np.mean(test_mean_tree), np.mean(test_mean_rnd), np.mean(test_mean_mlp)

np.mean(test_std) , np.mean(test_std_svc), np.mean(test_std_tree), np.mean(test_std_rnd), np.mean(test_std_mlp)

"""# Evaluation Metrics Calculations for Logisitic Regression"""

pipe_lr = pipe_lr.fit(X_train, y_train)

y_pred_train = pipe_lr.predict(X_train)

y_pred_test = pipe_lr.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_train, y_pred_train)

accuracy_score(y_test, y_pred_test)

from sklearn.metrics import classification_report

target_names = ['Normal', 'DoSattack', 'scan', 'malitiousControl', 'malitiousOperation', 'spying', 'dataProbing', 'wrongSetUp']

print(classification_report(y_train, y_pred_train, target_names=target_names))

print(classification_report(y_test, y_pred_test, target_names=target_names))

from sklearn.metrics import confusion_matrix
import itertools

cnf_matrix = confusion_matrix(y_test, y_pred_test)

for i in cnf_matrix:
    for j in i:
        print(j, end=' ')
    print()

"""# Evaluation Metrics Calculations for SVM"""

pipe_svc = pipe_svc.fit(X_train, y_train)
y_pred_train = pipe_svc.predict(X_train)
y_pred_test = pipe_svc.predict(X_test)

accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)

print(classification_report(y_train, y_pred_train, target_names=target_names))

print(classification_report(y_test, y_pred_test, target_names=target_names))

cnf_matrix = confusion_matrix(y_test, y_pred_test)
for i in cnf_matrix:
    for j in i:
        print(j, end=' ')
    print()

"""# Evaluation Metrics Calculations for Decision Tree"""

pipe_tree = pipe_tree.fit(X_train, y_train)
y_pred_train = pipe_tree.predict(X_train)
y_pred_test = pipe_tree.predict(X_test)

accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)

print(classification_report(y_train, y_pred_train, target_names=target_names))

print(classification_report(y_test, y_pred_test, target_names=target_names))

cnf_matrix = confusion_matrix(y_test, y_pred_test)
for i in cnf_matrix:
    for j in i:
        print(j, end=' ')
    print()

"""# Evaluation Metrics Calculations for Random Forest"""

pipe_rnd = pipe_rnd.fit(X_train, y_train)
y_pred_train = pipe_rnd.predict(X_train)
y_pred_test = pipe_rnd.predict(X_test)

y_pred_train = pipe_rnd.predict(X_train)
y_pred_test = pipe_rnd.predict(X_test)

accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)

print(classification_report(y_train, y_pred_train, target_names=target_names))

print(classification_report(y_test, y_pred_test, target_names=target_names))

cnf_matrix = confusion_matrix(y_test, y_pred_test)
for i in cnf_matrix:
    for j in i:
        print(j, end=' ')
    print()

"""# Evaluation Metrics Calculations for ANN"""

pipe_mlp = pipe_mlp.fit(X_train, y_train)
y_pred_train = pipe_mlp.predict(X_train)
y_pred_test = pipe_mlp.predict(X_test)

accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)

print(classification_report(y_train, y_pred_train, target_names=target_names))

import pickle

pickle.dump(pipe_mlp,open('MLP.sav', 'wb'))

print(classification_report(y_test, y_pred_test, target_names=target_names))

from sklearn.metrics import confusion_matrix
import itertools

cnf_matrix = confusion_matrix(y_test, y_pred_test)

cnf_matrix

for i in cnf_matrix:
    for j in i:
        print(j, end=' ')
    print()

"""#  CNN"""

import keras.models
import tensorflow
import numpy as np
from keras.layers.convolutional import Conv1D , Conv2D
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential    
from keras.optimizers import Adam  
from keras.layers import Dense, Activation, Flatten

# define model
n_steps = 11
n_features = 1
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# n_features = 1
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))

print(X_train.shape)
print(y_train.shape)

# y_train=np.ones((286352,10,64))
model.fit(X_train, y_train, epochs=10, verbose=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

y_test=np.ones((71589,10,64))
scores = model.evaluate(X_test, y_test)

print(scores)

"""# TUNING HYPERPARAMETERS IN ANN 

"""

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(35,), max_iter=10, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1)

pipe_mlp = Pipeline([('scl', StandardScaler()),('clf', mlp)])
train_sizes_mlp, train_scores_mlp, test_scores_mlp = learning_curve(estimator=pipe_mlp,X=X, y=y, train_sizes=np.linspace(0.2,1.0,5), cv=5, n_jobs=-1)
train_mean_mlp = np.mean(train_scores_mlp, axis=1)
train_std_mlp = np.std(train_scores_mlp, axis=1)
test_mean_mlp = np.mean(test_scores_mlp, axis=1)
test_std_mlp = np.std(test_scores_mlp, axis=1)

for i in train_mean_mlp:
    print(i)

for i in test_mean_mlp:
    print(i)

pipe_mlp = pipe_mlp.fit(X_train, y_train)
y_pred_train = pipe_mlp.predict(X_train)
y_pred_test = pipe_mlp.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)
# (0.9942273844778454, 0.994077302378857)-->base paper
# (0.992, 0.991)--->100 layers, adam
# (0.9942204000670504, 0.994091271005322)--->75 layers, sgd
# (0.9942238922724479, 0.994091271005322)---->100,sgd
# (0.9942169078616528, 0.994091271005322)---->35,sgd
# (0.9929597139185339, 0.9926385338529662)---->35,adam

from sklearn.metrics import classification_report
target_names = ['Normal', 'DoSattack', 'scan', 'malitiousControl', 'malitiousOperation', 'spying', 'dataProbing', 'wrongSetUp']
print(classification_report(y_train, y_pred_train, target_names=target_names))

import pickle

pickle.dump(pipe_mlp,open('MLP.sav', 'wb'))

print(classification_report(y_test, y_pred_test, target_names=target_names))

from sklearn.metrics import confusion_matrix
import itertools

cnf_matrix = confusion_matrix(y_test, y_pred_test)

cnf_matrix

for i in cnf_matrix:
    for j in i:
        print(j, end=' ')
    print()