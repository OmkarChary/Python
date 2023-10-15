import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
 
data = pd.read_csv("/Users/omkarchary/Documents/Python/1.01. Simple linear regression.csv")
 
X = data.iloc[:, :-1].values.reshape(-1,1)
y = data.iloc[:, -1].values.reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


clf = LinearRegression()


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(clf.coef_, clf.intercept_)
plt.plot(X_test, y_pred, color = "blue")

plt.show()


