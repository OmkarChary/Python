from seaborn import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_dataset('penguins')

print(data.head())
print(data.info())

data = data.dropna()

print(data.info())

data = data[data['species'] != 'Gentoo']

X = data[['bill_length_mm','bill_depth_mm']]
y = data['species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 24)

clf = SVC( kernel = 'linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test,y_pred))


import numpy as np

from seaborn import scatterplot

w = clf.coef_[0]

b = clf.intercept_[0]

x_visual = np.linspace(32, 57)

y_visual = -(w[0] / w[1]) * x_visual - b / w[1]

scatterplot(data = X_train, x = 'bill_length_mm', y = 'bill_depth_mm', hue = y_train)

plt.plot(x_visual, y_visual)
plt.show()