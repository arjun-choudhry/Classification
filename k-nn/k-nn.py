"""
@author: arjun
"""

# *********************************************************************************************************************
# AIM: Create a classification model to predict the purchasing activity of a group based on the given data using K-NN
# *********************************************************************************************************************

# Data-Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0)

# fitting the K-NN classification model
from sklearn.neighbors import KNeighborsClassifier
# metric = 'minkowski', p =2 are attributes required to instruct the model to follow Euclidean matrix
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)
classifier.fit(X_train,y_train)

# Predicting the values
y_pred = classifier.predict(X_test)

# Testing validity with confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)