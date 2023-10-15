from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

plt.gray()

plt.matshow(digits.images[0])

print(dir(digits))

print(digits.data[0])

#Create and train logistic regression model

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter = 3000)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.3, random_state = 34 )

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Measure Accuracy of the model 

print(clf.score(X_test, y_test))

print(clf.predict(X_test[0:5]))