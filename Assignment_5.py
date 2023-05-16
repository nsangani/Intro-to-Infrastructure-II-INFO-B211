# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:19:35 2020

@author: Sangani
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


#1
diabetes = pd.read_csv('C:\\Users\\Sangani\\Desktop\\Fall 2020\\Informatics\\Homework\\diabetes.csv', index_col=0, header=0)
data1 = diabetes.iloc[:,:-1]

print(diabetes.shape) # of rows and columns
print(len(diabetes)) # of rows
print(len(diabetes.columns)) # of columns
print(diabetes.columns) #all the features

print(diabetes['Outcome'].shape)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
x = diabetes[:615]  #80%
target = diabetes['Outcome']
y = target[153:] #20%

clf.fit(x,y)
print(diabetes[615:].shape)
predict = clf.predict(diabetes[615:])
print(predict)

y_pred = predict
y_true = target[615:]
""" 
In-class Headache w/ Xander
X = diabetes.drop('Outcome', axis=1)
y=['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
clf = svm.SVC()

clf.fit(X_train, y_train)
y_predict=clf.predict(X_test)
""" 

print(metrics.recall_score(y_true, y_pred))
print(metrics.accuracy_score(y_true,y_pred))
print(metrics.precision_score(y_true,y_pred))
print(metrics.f1_score(y_true, y_pred))

tp, tn, fp, fn = metrics.confusion_matrix(y_true, y_pred).ravel()
print(tp, tn, fp, fn)



#2
machine = pd.read_csv('C:\\Users\\Sangani\\Desktop\\Fall 2020\\Informatics\\Homework\\machine.csv')

print(machine.drop('ERP', axis=1).shape)

print(len(machine))
print(len(machine.columns))
print(machine['ERP'].shape)

target=machine['ERP']
machine = pd.read_csv('C:\\Users\\Sangani\\Desktop\\Fall 2020\\Informatics\\Homework\\machine1.csv')
clf = svm.SVR()

x = machine[:160]
y = target[:160]
print(x.shape)
print(y.shape)

clf.fit(x,y)
predict = clf.predict(machine[159:])
print(predict)

y_true = target[160:]
y_predict = predict

print(mean_squared_error(y_true, y_predict))
print(mean_absolute_error(y_true, y_predict))