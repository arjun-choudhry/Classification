"""
Created on Wed Dec 20 12:08:23 2017

@author: arjun
"""

# ************************************************************************************************************
# AIM: Create a classification model to predict the purchasing activity of a group based on the given data(with graph)  #
# ************************************************************************************************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
# If we want to specift certain columns, we have to write like this:  dataset.iloc[:,[1,2,3]]
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X[:,[0,1]] = sc_X.fit_transform(X[:,[0,1]])
# X = sc_X.inverse_transform(X)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

# Predicting the model on the test set
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

