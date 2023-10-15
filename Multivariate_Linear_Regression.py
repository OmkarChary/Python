# importing modules and packages 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing 

# importing data 
df = pd.read_csv('/Users/omkarchary/Documents/Python/Real-estate1.csv') 
df.drop('No', inplace=True, axis=1) 

print(df.head()) 

print(df.columns) 
print('dkjf')
print(df.shape[1])

# creating feature variables 
X = df.drop('Y house price of unit area', axis=1) 
y = df['Y house price of unit area'] 

print(X) 
print(y) 

# creating train and test sets 
X_train, X_test, y_train, y_test = train_test_split( 
	X, y, test_size=0.3, random_state=101) 

# creating a regression model 
model = LinearRegression() 

# fitting the model 
model.fit(X_train, y_train) 

# making predictions 
predictions = model.predict(X_test) 

# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, predictions)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions)) 

# coefficients of the Model
print('Coefficients : {}'.format(model.coef_))
