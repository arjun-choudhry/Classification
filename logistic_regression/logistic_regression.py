"""
@author: arjun
"""

# ************************************************************************************************************
# AIM: Create a classification model to predict the purchasing activity of a group based on the given data   #
# ************************************************************************************************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
#If we want to specift certain columns, we have to write like this:  dataset.iloc[:,[1,2,3]]
X = dataset.iloc[:,1:4].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lblEncoder_sex = LabelEncoder()
X[:,0] = lblEncoder_sex.fit_transform(X[:,0])
ohe = OneHotEncoder(categorical_features = [0])
X = ohe.fit_transform(X).toarray()

#removing one dummy variable to avoid the dummy variable trap
X = X[:,1:]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X[:,[1,2]] = sc_X.fit_transform(X[:,[1,2]])
# X = sc_X.inverse_transform(X)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)