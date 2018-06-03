#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:58:52 2018

@author: Shashank Pawar
"""

# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting the Regression Model to the dataset
from sklearn.svm import SVR
reg1 = SVR(kernel='rbf')
reg1.fit(X,y)

reg2 = SVR(kernel='linear')
reg2.fit(X,y)

reg3 = SVR(kernel='sigmoid')
reg3.fit(X,y)

reg4 = SVR(kernel='poly')
reg4.fit(X,y)

# Predicting a new result
y1 = sc_y.inverse_transform(reg1.predict(sc_X.transform(np.array([[6.5]]))))
y2 = sc_y.inverse_transform(reg2.predict(sc_X.transform(np.array([[6.5]]))))
y3 = sc_y.inverse_transform(reg3.predict(sc_X.transform(np.array([[6.5]]))))
y4 = sc_y.inverse_transform(reg4.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, reg1.predict(X), color = 'blue')
plt.plot(X, reg2.predict(X), color = 'red')
plt.plot(X, reg3.predict(X), color = 'cyan')
plt.plot(X, reg4.predict(X), color = 'green')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
