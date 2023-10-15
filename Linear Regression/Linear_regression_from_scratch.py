import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X = 2 * np.random.rand(100,1)
Y = 5 + 3 * X + np.random.randn(100,1)




#plt.scatter(data.SAT, data.GPA)
#plt.show()


    
def gradient_decent(w1_cur, w0_cur, x, y, L_rate):
    w1_grad = 0
    w0_grad = 0
    
    n = len(X)
    
    for i in range(n):
        x = X[i]
        y = Y[i]
        y_cap = w1_cur * x + w0_cur
        w1_grad +=  -(2/n) * x * (y - y_cap)
        w0_grad += -(2/n) * (y - y_cap)
        
    w1 = w1_cur - w1_grad * L_rate
    w0 = w0_cur - w0_grad * L_rate
    return w1, w0

w1 = 0 
w0 = 0


L_rate = 0.01
epochs = 1000

for i in range(epochs):
    w1, w0 = gradient_decent(w1, w0, X, Y, L_rate)
    
    
print(w1, w0)

plt.scatter(X, Y, color = "black")
plt.plot(X, [w1 * X[i] + w0 for i in range(len(X))], color = "red")
plt.show()