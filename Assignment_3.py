# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:57:22 2020

@author: Sangani
"""

# Question #1

import pandas as pd

df = pd.read_csv('forestfires.csv', index_col=0)
print(df.shape)
print(df.size)
print(df.describe())
df1 = df.head(5)
print(df1)
df2 = (df.tail(5))
print(df2)

print('\n')
df_row_reindex = pd.concat([df1,df2], ignore_index=True)
print(df_row_reindex)

print('\n')

df_merge = pd.merge(df1, df2, on='X')
print(df_merge)
print('\n')

print(df[df['temp']> 20])
print('\n')
print(df.drop_duplicates(subset=['month', 'day']))

print(df.info())

import numpy as np

np_array = df[['Y','month','day']].to_numpy()
print(np_array)

rd = df.sample(n=50)
rd.insert(0, 'ID', range(1,51))
print(rd)


sum = df['FFMC'] + df['DMC']
print(sum)

substract = df['DC'] - df['DMC']
print(substract)

square = df['RH']**2
print(square)

area_sum = df.agg({'area':['sum']})
print(area_sum)

print(df['wind'].value_counts())

print(df.iloc[[11]]) # 10th sample in the dataset

print(df.isnull().values.any()) # No null values

# Question #2

Cars = pd.DataFrame({'Brand': ['Hunda','Ford', 'Hundi', 'Toyota'],
                    'Price': [23000,28000,7000,26000]})
print(Cars)

print(Cars.to_csv('Cars_price.csv', index = False))











