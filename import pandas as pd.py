import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
 
data = pd.read_csv("/Users/omkarchary/Documents/Python/1.01. Simple linear regression.csv")
 
X = data['SAT']
y = data['GPA']

""" X = X.to_numpy(X)
y = y.to_numpy(y) """


plt.scatter(X, y, color = "red")
plt.xlabel("Year")
plt.ylabel("GPA")

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.3, random_state = 24)

plt.show()
clf = LinearRegression()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

plt.plot(X_train, y_pred, color = "blue")

plt.show()

