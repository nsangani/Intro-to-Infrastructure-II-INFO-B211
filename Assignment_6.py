# -*- coding: utf-8 -*-
"""
@author: Sangani
"""
#--------------------------------------------------------------------
# Import Library
#--------------------------------------------------------------------
import pandas as pd
import numpy as np
pip install(seaborn)
import seaborn as sns
import matplotlib.pyplot as plt 

#--------------------------------------------------------------------
# Import Dataset
# Store column data in specific variables
#--------------------------------------------------------------------
iris = pd.read_csv('iris_data.csv')
print(iris.head())

sepal_length = iris['sepal_length']
sepal_width = iris['sepal_width']
petal_length = iris['sepal_length']
petal_width = iris['petal_width']
species = iris['class']

species_unique = pd.unique(iris['class'])
print(species_unique)

iris_setosa = iris.loc[iris["class"]=="Iris-setosa"]
iris_virginica = iris.loc[iris["class"]=="Iris-virginica"]
iris_versicolor = iris.loc[iris["class"]=="Iris-versicolor"]

#--------------------------------------------------------------------
# Question 1.a > Look at the distributions for each of the traits 
# (sepal length, sepal width, petallength, and petal width). 
#--------------------------------------------------------------------


sns.histplot(data = sepal_length)
plt.title('Sepal Length Distribution')
plt.savefig('dist_sepal_length.png')
plt.show()

sns.histplot(data = sepal_width)
plt.title('Sepal Width Distribution')
plt.savefig('dist_sepal_width.png')
plt.show()

sns.histplot(data = petal_length)
plt.title('Petal Length Distribution')
plt.savefig('dist_petal_length.png')
plt.show()

sns.histplot(data = petal_width)
plt.title('Petal Width Distribution')
plt.savefig('dist_petal_width.png')
plt.show()

"""
#Combined Comparison via 'kde' (Kernel Density Estimation)

iris = sns.load_dataset('iris')
sns.kdeplot(data=iris)
plt.title('Density Distribution of Iris Traits')
plt.savefig('kde_Iris_trait.png')
plt.show()
"""

#--------------------------------------------------------------------
# Question 1.b > Create a categorical plot for each trait by species. 
# Boxplot
#--------------------------------------------------------------------

# !!! NOTE: Change x = 'class' to 'species' if it doesn't detect class !!! 

sns.boxplot(x='class', y='sepal_length', data=iris)
plt.title('Sepal Length Dist. Across Species')
plt.savefig('cat_sepal_length.png')
plt.show()

sns.boxplot(x='class', y='sepal_width', data=iris)
plt.title('Sepal Width Dist. Across Species')
plt.savefig('cat_sepal_width.png')
plt.show()

sns.boxplot(x='class', y='petal_length', data=iris)
plt.title('Petal Length Dist. Across Species')
plt.savefig('cat_petal_length.png')
plt.show()

sns.boxplot(x='class', y='petal_width', data=iris)
plt.title('Petal Width Dist. Across Species')
plt.savefig('cat_petal_width.png')
plt.show()


#--------------------------------------------------------------------
# Question 1.c > Create relational plots between trait pairs 
# (sepal length vs sepal width) & (petal length vs petal width)
# include a way of distinguishing by species. 
# Scatterplot
#--------------------------------------------------------------------

# !!! NOTE: Change hue = 'class' to 'species' if it doesn't detect class !!!  

sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue ="class")
plt.title('Sepal Legnth vs Sepal Width')
plt.savefig('sepal_length_vs_sepal_width.png')
plt.show()

sns.scatterplot(data=iris, x='petal_length', y='petal_width', hue ="class")
plt.title('Petal Legnth vs Petal Width')
plt.savefig('petal_length_vs_petal_width.png')
plt.show()


#--------------------------------------------------------------------
# Question 1.d > Based upon your analysis, which species are most related? 
# Support your claim. 
#--------------------------------------------------------------------

# > Both, Versicolor Species & Virginica have longer and wider petal length and width,
#   respectively, compare to setosa as shown in the distribution. Therefore, 
#   I can say that Versicolor species and virginica is more closely related. 
#   Moreover, Sepal length and width relational plot also shows a clear overlap
#   between these two closely related species and setosa in its distict cluster. 


#--------------------------------------------------------------------
# Question 2.a > Create a heatmap for the participants, workout time, and pulse. 
# (Hint use the pivot command from Pandas to get the data in appropriate format) 
#--------------------------------------------------------------------

data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/exercise.csv')
print(data.head())

participants = data.pivot_table(index='id', columns='time', values='pulse')
print(participants)

ax = sns.heatmap(participants)
plt.ylabel('Participants')
plt.title('Pulse of local Gym Members')
plt.savefig('Heatmap.png')
plt.show()

#--------------------------------------------------------------------
# Question 2.b > Create a categorical plot for (pulse by diet) & 
# (pulse by type of exercise). 
#--------------------------------------------------------------------

sns.boxplot(x='diet', y='pulse', data=data)
plt.title('Pulse by Diet')
plt.savefig('Pulse_by_Diet.png')
plt.show()


sns.boxplot(x='kind', y='pulse', data=data)
plt.title('Pulse by Type of Excercise')
plt.savefig('Pulse_by_Excercise.png')
plt.show()

#--------------------------------------------------------------------
# Question 2.c > Give a brief explanation of the meaning of the 3 graphs to the elementary students. 
#--------------------------------------------------------------------

# > In general, gym goers have pulse rate on average around 92 when resting, 
#   95 when walking, and 110 when running. Moreoover, we can likely say that 
#   people with no fat tend to do longer and extensive workout then with low fat
#   therefore they have higher pulse rate of 100 on average compare to 
#   low fat individuals, 95. And few of these individuals, around 4-6, reached 
#   the pulse rate of 120 and over after 15 minutes into their workout. 




