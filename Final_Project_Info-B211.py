# -*- coding: utf-8 -*-
"""
@author: Sangani
"""

"""
#-----------------------------------------------------------------------------
#                  Description of Features in Heart dataset
#-----------------------------------------------------------------------------

1. Age 
2. Sex
    --Value 1: male 
    --Value 0: female
3. Chest Pain: value range 0-3. The higher the number, the lesser are the odds of heart attack.
    -- Value 0: asymptomatic (Disease with no symptoms)
    -- Value 1: atypical angina (Experiences nausea or shortness of breath) 
    -- Value 2: non-anginal pain 
    -- Value 3: typical angina (chest discomfort)
4. Resting blood pressure: normal pressure with no exercise in mmHg
5. Cholesterol: Blockage for blood supply in the blood vessels in mg/dl
6. Fasting Blood Pressure: Blood sugar taken after a long gap between a meal and the test. 
    -- Value 0: no blood suger not higher than 120 mg/dl
    -- Value 1: yes blood suger is higher than 120 mg/dl
7. Rest ECG results: ECG values taken while on rest 
    -- Value 0: probable or definite left ventricular hypertrophy - thickening of wall
    -- Value 1: normal
    -- Value 2: having ST-T wave abnormality 
8. The Maximum Heart Rate under stress test
9. Exercise induced angina: chest pain while exercising or doing any physical activity.
    -- Value 0: no
    -- Value 1: yes
10. ST Depression: difference between value of ECG at rest and after exercise.
11. ST Slope: the tangent to the depression 
    ---Value. 0: desecnding 
    ---Value 1: flat
    ---Value 2: ascending
12. The number of major blood vessels supplying blood to heart blocked. Range 0-4
13. The Types of thalassemia: 
    -- Value 1 = fixed defect (no blood flow in some part of the heart)
    -- Value 2 = normal 
    -- Value 3 = reversable defect (a blood flow is observed but it is not normal)
14. Heart attack prediction: 
    -- Value 0: Heart attack
    -- Value 1: No Heart attack 

# Continuous Featuers: Age, Resting bp, Chol, Max_heart Rate.
# Discrete Features: Sex, Chest Pain, Fasting Blood Pressure, ECG Results, Excercise Induced Angina,
#                    ST Depression, ST Slope, # of blood vessels, Thalassemia, Heart Attack

"""


#-----------------------------------------------------------------------------
#                        Importing Libraries

#-----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import scikitplot as skplt
import warnings
warnings.filterwarnings("ignore")
# Note: SciPy is used by sklearn to compute the prediction model

#-----------------------------------------------------------------------------
#                      Importing Dataset and Renaming 
#-----------------------------------------------------------------------------

'''
age: Age
sex: Sex
cp: Chest_pain
trestbps: Resting_blood_pressure
chol: Cholesterol
fbs: Fasting_blood_sugar
restecg: ECG_results
thalach: Maximum_heart_rate
exang: Exercise_induced_angina
oldpeak: ST_depression
slope: ST_slope
ca: Major_vessels
thal: Thalassemia_types
target: Heart_attack
'''

df = pd.read_csv('heart.csv')
df.rename(columns = {'age':'Age','sex':'Sex','cp':'Chest_pain','trestbps':'Resting_blood_pressure',
                     'chol':'Cholesterol','fbs':'Fasting_blood_sugar','restecg':'ECG_results',
                     'thalach':'Maximum_heart_rate','exang':'Exercise_induced_angina',
                     'oldpeak':'ST_depression','ca':'Major_vessels',
                   'thal':'Thalassemia_types','target':'Heart_attack',
                   'slope':'ST_slope'}, inplace = True)

#-----------------------------------------------------------------------------
# Assign Sex column value 1: Male and 0: Female for visualization purpose
# And Heart Attack column value 0: Yes and 1: No 
#-----------------------------------------------------------------------------

df['Sex'].replace({1:'Male',0:'Female'}, inplace = True)
df['Heart_attack'].replace({1:'Heart_attack - No',0:'Heart_attack - Yes'}, inplace = True)

#-----------------------------------------------------------------------------
# Excercise Induced Angina(EIA)  has 0 for no pain and 1 for pain whereas
# Chest pain (CP) column value has decending pain with 0 being most critical and 3 being least
# Thus, to correlate EIA and CP, EIA values are swap: 0s from No to Yes and same for 1 values 
#-----------------------------------------------------------------------------

df['Exercise_induced_angina'].replace({1:'No',0:'Yes'}, inplace = True)
df['Exercise_induced_angina'].replace({'No': 0,'Yes': 1}, inplace = True)

#-----------------------------------------------------------------------------
# Check for Null Data Values and basic stat about the dataset - Shape and Attributes
#-----------------------------------------------------------------------------

print('Null Values in Dataset: ', df.isnull().sum().sum() , '\n') # Looks for any null values in the dataset
print(df.info() , '\n')
print(df.shape , '\n')
print(df.describe() , '\n')

#-------------------------------------------------------------------------
# Basic Gender and Heart Attack graphs [Countplot, PieChart]
#-------------------------------------------------------------------------

ax = sns.countplot(df.Sex) # More Males than Females
plt.show()

gender = df['Sex']
label_count = gender.value_counts()
plt.pie(label_count, labels=['Male','Female'], 
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

sns.countplot(df.Heart_attack) # Approx. Similar Spread
plt.show()

Heart_attack = df['Heart_attack']
label_count = Heart_attack.value_counts()
plt.pie(label_count, labels=['Heart_attack - No','Heart_attack - Yes'], 
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

#-------------------------------------------------------------------------
#                          Correlation Heatplot
#-------------------------------------------------------------------------

sns.set(font_scale=1.1)
correlation_train = df.corr()
mask = np.triu(correlation_train.corr())
plt.figure(figsize=(20, 12))
sns.heatmap(correlation_train,
            annot=True,
            fmt='.3f',
            cmap=sns.diverging_palette(230, 20, as_cmap=True),
            linewidths=1,
            cbar=True)

plt.show()

'''
Following shows higher correlation from the Heatplot above:
    Age vs Max_Heart_rate
    Chest Pain vs Excercise_Induced Angina
    ST_slope vs Max_Heart_Rate
    Chest Pain vs Max_Heart_Rate
'''

#-----------------------------------------------------------------------------
# Mean and Standard Deviation of Continuous Data
#-----------------------------------------------------------------------------

df.groupby(['Sex', 'Heart_attack'])[['Age','Resting_blood_pressure','Cholesterol',
                                      'Maximum_heart_rate']].agg([np.mean, np.std])

#-----------------------------------------------------------------------------
#  ScatterPlot [May or May not use these Graphs in the Presentation]
#  Age vs Max_Heart_rate  [Different Ways Correlated]
#-----------------------------------------------------------------------------

#1
df['Age'].plot.hist(bins = 15, xlabel = 'Age', title = 'Age Distribution')

#2
plt.figure(figsize=(8,8))
sns.scatterplot(x=df['Age'],y=df['Maximum_heart_rate'],hue=df['Heart_attack'])
plt.xlabel('Age')
plt.ylabel('Maximum_heart_rate')
plt.title('Age vs Max_Heart_rate')
plt.show()

#3
sns.scatterplot(df.Age[df.Heart_attack=='Heart_attack - Yes'], y=df.Maximum_heart_rate[(df.Heart_attack=='Heart_attack - Yes')], color='red')
sns.scatterplot(df.Age[df.Heart_attack=='Heart_attack - No'], y=df.Maximum_heart_rate[(df.Heart_attack=='Heart_attack - No')], color='green')
plt.legend(["Heart_attack Yes", "Heart_attack No"])
plt.title('Age vs Max_Heart_rate')

#4
sns.relplot(x ='Age', y ='Maximum_heart_rate', col = 'Sex', data = df, color = 'turquoise')

#-----------------------------------------------------------------------------
# line graphs for clear correlation: [May or May not use these Graphs in the Presentation]
#-----------------------------------------------------------------------------
'''
    Age vs Maximun_heart_rate
    Age vs Resting_blood_pressure
    Age vs Cholesterol
    Chest_pain vs Excercise_Induced_Angina
    Chest_pain vs Maximum_heart_rate
    ST_slope vs Maximum_heart_rate
    Chest_pain vs Excercise_induced_pain
'''
    
sns.relplot(x = 'Age', y = 'Maximum_heart_rate', col = 'Sex', kind = 'line', data=df,aspect = 1,height = 7, color = 'green')

sns.relplot(x = 'Age', y = 'Resting_blood_pressure', col = 'Sex', kind = 'line', data=df,aspect = 1,height = 7, color = 'red')

sns.relplot(x = 'Age', y = 'Cholesterol', col = 'Sex', kind = 'line', data=df,aspect = 1,height = 7, color = 'purple')

sns.relplot(x = 'Chest_pain', y = 'Exercise_induced_angina', col = 'Sex', kind = 'line', data=df,aspect = 1,height = 7, color = 'orange')

sns.relplot(x = 'Chest_pain', y = 'Maximum_heart_rate', col = 'Sex', kind = 'line', data=df,aspect = 1,height = 7, color = 'teal')

sns.relplot(x = 'ST_slope', y = 'Maximum_heart_rate', col = 'Sex', kind = 'line', data=df,aspect = 1,height = 7, color = 'yellow')

sns.relplot(x = 'Chest_pain', y = 'Exercise_induced_angina', col = 'Sex', kind = 'line', data=df,aspect = 1,height = 7, color = 'blue')
plt.show()
#-----------------------------------------------------------------------------
# Count Plots: target vs sex, chest pain(cp), age, fasting blood pressure(fbp),
#                        resting ECG (restecg), Excercise induced Angina(exang),
#                        Blocked Blood Vessels (ca), Thalassemia (thal)
# [May or May not use these Graphs in the Presentation]
#-----------------------------------------------------------------------------

sns.countplot(df.Sex,hue=df.Heart_attack)  # More Males have Heart Attack than Female
plt.show()

sns.catplot(x ='Age', y ='Heart_attack', col = 'Sex', data = df, color = 'purple', kind = 'box')
plt.show()

sns.countplot(df.ECG_results,hue=df.Heart_attack) # Value 0 probable or definite left ventricular hypertrophy
plt.show()

sns.countplot(df.Chest_pain,hue=df.Heart_attack) # Nothing Substantive 
plt.show()

sns.countplot(df.Fasting_blood_sugar,hue=df.Heart_attack) # No direct correlation
plt.show()

sns.countplot(df.Exercise_induced_angina,hue=df.Heart_attack) # Excercise induced Angina has higher HA rate
plt.show()

sns.countplot(df.Major_vessels,hue=df.Heart_attack) # 1 vessel and more has higher HA rate
plt.show()

sns.countplot(df.Thalassemia_types,hue=df.Heart_attack)
plt.show()

#-----------------------------------------------------------------------------
# Density Plot: Heart Attack [0,1] vs [Continous Data] Age, Resting_Blood_Pressure,
#               Cholesterol, and Max_Heart_Rate
# Checking for Normal Distribution before using data for Classifiers
# [May or May not use these Graphs in the Presentation]
#-----------------------------------------------------------------------------

sns.distplot(df.Age[df.Heart_attack=='Heart_attack - No'])
sns.distplot(df.Age[df.Heart_attack=='Heart_attack - Yes'])
plt.legend(['Heart_attack - No','Heart_attack - Yes'])
plt.show()

sns.distplot(df.Resting_blood_pressure[df.Heart_attack=='Heart_attack - No'])
sns.distplot(df.Resting_blood_pressure[df.Heart_attack=='Heart_attack - Yes'])
plt.legend(['Heart_attack - No','Heart_attack - Yes'])
plt.show()

sns.distplot(df.Cholesterol[df.Heart_attack=='Heart_attack - No'])
sns.distplot(df.Cholesterol[df.Heart_attack=='Heart_attack - Yes'])
plt.legend(['Heart_attack - No','Heart_attack - Yes'])
plt.show()

sns.distplot(df.Maximum_heart_rate[df.Heart_attack=='Heart_attack - No'])
sns.distplot(df.Maximum_heart_rate[df.Heart_attack=='Heart_attack - Yes'])
plt.legend(['Heart_attack - No','Heart_attack - Yes'])
plt.show()

#-----------------------------------------------------------------------------
# Graphical Analysis Description
#-----------------------------------------------------------------------------

print('Females tend to suffer less from heart attack and men have a higher chances of getting a heart attack.', '\n',
"Men don't have any defined age span that they are safe from heart attack.", '\n',
'Men below 30 also suffered from heart attack unlike women. Thus, men needs to be','\n',
 'more careful with their health conditions and situations.','\n')

#-----------------------------------------------------------------------------
# Machine Learning Classifiers/Models
#-----------------------------------------------------------------------------

df['Sex'].replace({'Male':1, 'Female': 0}, inplace = True)
df['Heart_attack'].replace({'Heart_attack - No':1, 'Heart_attack - Yes':0}, inplace = True)

X = df.drop(['Heart_attack'], axis = 1)
# X = X.to_numpy()
y = df['Heart_attack']
# y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)

print('Training data : {},{} '.format(X_train.shape, y_train.shape))
print('Testing data : {},{} '.format(X_test.shape, y_test.shape))

#-----------------------------------------------------------------------------
# Random Forest Classifier
#-----------------------------------------------------------------------------

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred_rf = clf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)

print('Random Forest Classifier Prediction')
print('Confusion Matrix: ', '\n' , cm_rf) 
print('Recall: ', metrics.recall_score(y_test, y_pred_rf))
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_rf))
print('Precision: ', metrics.precision_score(y_test, y_pred_rf))
print('f1_score: ', metrics.f1_score(y_test, y_pred_rf))

rfc = classification_report(y_test,y_pred_rf)
print(rfc)

pred = clf.predict(X_test)
matrix = (y_test,pred)
skplt.metrics.plot_confusion_matrix(y_test, pred,figsize=(10,5),title='Confusion Matrix: Random Forest Classifier')

#-----------------------------------------------------------------------------
# Support Vector Machine Classifier
#-----------------------------------------------------------------------------

clf = svm.SVC() 
clf.fit(X_train,y_train)
y_pred_rf = clf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)

print('Support Vector Machine Classifier Prediction')
print('Confusion Matrix: ', '\n' , cm_rf) 
print('Recall: ', metrics.recall_score(y_test, y_pred_rf))
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_rf))
print('Precision: ', metrics.precision_score(y_test, y_pred_rf))
print('f1_score: ', metrics.f1_score(y_test, y_pred_rf))

rfc = classification_report(y_test,y_pred_rf)
print(rfc)

pred = clf.predict(X_test)
matrix = (y_test,pred)
skplt.metrics.plot_confusion_matrix(y_test, pred,figsize=(10,5),title='Confusion Matrix: Support Vector Machine Classifier')

#-----------------------------------------------------------------------------
# Logistic Regression Classifier
#-----------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred_rf = clf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)

print('Logistic Regression Classifier Prediction')
print('Confusion Matrix: ', '\n' , cm_rf) 
print('Recall: ', metrics.recall_score(y_test, y_pred_rf))
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_rf))
print('Precision: ', metrics.precision_score(y_test, y_pred_rf))
print('f1_score: ', metrics.f1_score(y_test, y_pred_rf))

rfc = classification_report(y_test,y_pred_rf)
print(rfc)

pred = clf.predict(X_test)
matrix = (y_test,pred)
skplt.metrics.plot_confusion_matrix(y_test, pred,figsize=(10,5), title='Confusion Matrix: Logistic Regression Classifier')

#-----------------------------------------------------------------------------
# Classifier Comparison Description
#-----------------------------------------------------------------------------

print('Logistic Regression shows higher Recall, Accuracy, and f1_score than Random Forest and SVM.', '\n',
      'And Random Forest has higher precision than Logistic Regression. In terms of ranking,', '\n',
      'Logistic Repression is better than Random Forest. The least effective model in this case is SVM.', '\n')


print('After changing dataset to numpy and redone the prediction,', '\n',
      'prediction of all the classifier increased')







