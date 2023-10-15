import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("/Users/omkarchary/Documents/Python/polynomial-regression.csv")

print(data)
X = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values.reshape(-1,1)


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 4, include_bias = False)

X_poly = poly_features.fit_transform(X)

from sklearn.linear_model import LinearRegression

clf = LinearRegression()

clf.fit(X_poly, y)

y_pred = clf.predict(X_poly)

plt.scatter(X, y, c = "r")

plt.xlabel("araba_fiyat")
plt.ylabel("araba_max_hiz")
plt.plot(X, y_pred, c = "b")

plt.show()