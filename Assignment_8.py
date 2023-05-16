# -*- coding: utf-8 -*-
"""

@author: Sangani
"""

import pandas as pd

df = pd.read_csv('spam_ham_dataset.csv')
print(df.columns)
print('Shape: ', df.shape)
print('\n')

print('0 = Not Spam & 1 = Spam [Email]')
print(df['label_num'].value_counts())

import matplotlib.pyplot as plt

label = df['label_num']
label_count = label.value_counts()
plt.bar(label_count.index, label_count)
plt.xlabel('0 = Not Spam  $  1 = Spam')
plt.ylabel('No. of Email')
plt.title('Spam vs Not Spam')
plt.show()

ax = ['Not Spam', 'Spam']
ay = [3672,1499]
plt.bar(ax,ay)
plt.xlabel('Email Type')
plt.ylabel('No. of Emails')
plt.title('Spam vs Not Spam')
plt.show()

import nltk
from nltk.tokenize import word_tokenize

sample_0 = df.iloc[141,2] #140th index and 3rd column
# print(sample_0)

text0 = word_tokenize(sample_0)
# print(text0)
print('\n')


print('Total Words from subject 140: ', len(text0))
print('\n')

from nltk.corpus import stopwords

a = set(stopwords.words('english'))
sample_1 = df.iloc[1266,2] #1265th index and 3rd column
text1 = word_tokenize(sample_1.lower())
# print(text1)

stopwords = [x for x in text1 if x in a]
print(stopwords)
print('Total Stopwords from subject 1265: ', len(stopwords))
print('\n')

filtered_text1 = [x for x in text1 if x not in a]
print(filtered_text1)
print('Total Filtered Words from subject 1265:: ', len(filtered_text1))
print('\n')

from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

sample_2 = df.iloc[1836,2] #1835th index and 3rd column
text2 = word_tokenize(sample_2.lower())
# print(text2)
filtered_text2 = [x for x in text1 if x not in a]

similar = []
different = []
count1 = 0
for word in filtered_text2:
    pst = PorterStemmer()
    stem = pst.stem(word)
    
    lem = WordNetLemmatizer()
    lemma = lem.lemmatize(word)
    
    if stem == lemma:  #Check for similar output
        similar.append(stem)
    
   
    if stem != lemma:
        different.append(stem + ':' + lemma)
print('Stem vs Lemmatization')    
print('Similar Output:', similar)
print('Similar Output:', len(similar))
print('Different Output:',different)    
print('Different Output:', len(different))
print('\n')

sample_3 = df.iloc[5012,2] #5011th index and 3rd column
text3 = word_tokenize(sample_3.lower())
# print(text3)
# filtered_text3 = [x for x in text1 if x not in a] 

postag_words= nltk.pos_tag(text3)
print(postag_words)
for tag in postag_words:
    nltk.help.upenn_tagset(tag[1])
    print('\n')


from sklearn.feature_extraction.text import CountVectorizer

print(df.head())
print('\n')
print(df.info())
print('\n')
#Generating Features using Bag of Words Approach

print('Using Bag of Words Approach: ')

cv = CountVectorizer(max_features=50447) #1000 most frequent words to choose as feature
text_counts= cv.fit_transform(df['text']) 
print(text_counts.shape)

from sklearn.model_selection import train_test_split
X = text_counts
y= df['label_num']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred_rf = clf.predict(X_test)
                
from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_test, y_pred_rf)
print('Confusion Matrix: ' , cm_rf)

from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

 
print('Recall: ', metrics.recall_score(y_test, y_pred_rf))
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_rf))
print('Precision: ', metrics.precision_score(y_test, y_pred_rf))
print('f1_score: ', metrics.f1_score(y_test, y_pred_rf))

print('\n')
#---------------------------------------------------------------------------
print('Using TF-IDF Approach: ')

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
text_tf = tf.fit_transform(df['text'])

print(text_tf.shape)

from sklearn.model_selection import train_test_split
X = text_tf
y= df['label_num']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2) #20% test and 80% train

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred_rf = clf.predict(X_test)
                
from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_test, y_pred_rf)
print('Confusion Matrix: ' , cm_rf)

from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

 
print('Recall: ', metrics.recall_score(y_test, y_pred_rf))
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_rf))
print('Precision: ', metrics.precision_score(y_test, y_pred_rf))
print('f1_score: ', metrics.f1_score(y_test, y_pred_rf))
print('\n')

print('Performance Comparison: ')
print('With equal number of feature for both Approach,' + '\n' + ' I am getting higher Recall and f1_score in TF_IDF Approach and' + '\n' + 'higher Accuracy and Precision in Bag of words Approach.')






















