import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/omkarchary/Documents/Python/framingham.csv")

print(data.head())

print(data.shape)

data.drop('education', axis = 1, inplace = True)

data.info()


#Filling missing data 
data['cigsPerDay'].fillna(value = 0.0, inplace = True)
data['BPMeds'].fillna(value = (data['BPMeds'].mean()), inplace = True)
data['totChol'].fillna(value = (data['totChol'].mean()), inplace = True)
data['BMI'].fillna(value = (data['BMI'].mean()), inplace = True )
data['glucose'].fillna(value = (data['glucose'].mean()), inplace = True )

data = data.dropna()
print(data.isnull().sum())

X = data.drop('TenYearCHD', axis = 1)
y = data['TenYearCHD']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 35 )

#Scaling the data
SS = StandardScaler()

SS.fit(X_train)

X_train_scaled = SS.transform(X_train)
X_test_scaled = SS.transform(X_test)

#Creating the model
clf = LogisticRegression()

clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

from sklearn.metrics import accuracy_score

print('Accuracy : {}'.format(accuracy_score(y_pred,y_test)))