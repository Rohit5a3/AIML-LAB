# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 22:35:37 2023

@author: Dell
"""

import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

#Load csv
data= pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\AIML\\6\p-tennis.csv")
print("The first 5 values od data is:\n",data.head())

#obtain train data and output
X=data.iloc[:,:-1]
print("The first 5 values of train data:\n",X.head())

y=data.iloc[:,-1]
print("The first 5 values of train data:\n",y.head())

#convert into numbers
le_outlook=LabelEncoder()
X.Outlook=le_outlook.fit_transform(X.Outlook)

le_Temperature=LabelEncoder()
X.Temperature=le_Temperature.fit_transform(X.Temperature)

le_Humidity=LabelEncoder()
X.Humidity=le_Humidity.fit_transform(X.Humidity)

le_Windy=LabelEncoder()
X.Windy=le_Windy.fit_transform(X.Windy)

print("\n Now the train data is:\n",X.head())

le_PlayTennis=LabelEncoder()
y=le_PlayTennis.fit_transform(y)

print("\n Now the train data is:\n",y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy score:\n")
print(accuracy_score(y_test,y_pred))