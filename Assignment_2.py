# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:55:07 2020

@author: Sangani
"""

import numpy as np

#Problem 1

a = np.array([[10,1,3],[3,5,12],[7,13,5]])
print(np.all(a!=5))

#Problem 2

b = np.array([5,3,8,7,10])
b = b.astype(float)
print(b)

#problem 3
import random 
c = np.random.rand(1000)
print(np.mean(c))
print(np.std(c))

#problem 4

d = np.array([[0,1,3],[3,7,12],[6,13,21]])
print(d.cumsum(axis=1))  #rows
print(d.cumsum(axis=0))  #columns

#problem 5

e = np.array([10,30,50,70,100,80])
print(np.diff(e))

# problem 6

months = np.arange('2017-01', '2018-01', dtype='datetime64[M]')
print(months)

"""
['2017-01' '2017-02' '2017-03' '2017-04' '2017-05' '2017-06' '2017-07'
 '2017-08' '2017-09' '2017-10' '2017-11' '2017-12']
"""
#problem 7

Iris = open('iris.data.csv')
array = np.genfromtxt(Iris, delimiter=',',names=True)
# array_header = np.genfromtxt(Iris, delimiter=',')
print(array.shape) #(150,)
print(array.dtype.names) #('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class')
print(array.dtype.names[-1]) #class
print(array.dtype[0],array.dtype[1],array.dtype[2],array.dtype[3])
print(array.dtype[-1])
new_array = array[:,-1]
print(new_array.shape)
last_column = array[-1]
print(last_column.shape)
print(array[24,1]) #25th data/row and 2nd column 
print(array[98,-1]) #99th row and last column