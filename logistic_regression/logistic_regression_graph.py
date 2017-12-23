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

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
# The below line allows us to create the grid of the plot that we want to plot. So, we take the minimum of the age column-1, maximum of age column +1 and give the step configuration of 0.01. Hence, using this we create the pixels along the AGE axis that we want in the plot. We do the same for the Salary axis. These values are then assigned to X1 and X2.
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()