import numpy as np

X1 = 2 * np.random.rand(100,1)
X2 = 3 * np.random.rand(100,1)
Y = 5 + 3 * X1 + 4 * X2 + np.random.randn(100,1)


def fit(X1, X2, Y, lr, w1, w2, b, cost):
    dw1 = 0
    dw2 = 0
    db = 0
    
    n = len(X1)
    for i in range(n):
        
        x1 = X1[i]
        x2 = X2[i]
        y = Y[i]
        
        y_cap = x2*w2 + x1*w1 + b
        
        cost += (1/n) * (y_cap - y)**2
        
        dw2 +=  -(2/n) * (y - y_cap) * x2
        dw1 += -(2/n) * (y - y_cap) * x1
        db += -(2/n) * (y - y_cap)
        
    w2_new = w2 - lr * dw2
    w1_new = w1 - lr * dw1
    b_new = b - lr * db
    print("Cost : {}".format(cost))
    return w2_new, w1_new, b_new


w1 = 0
w2 = 0
b = 0 
n_iter = 1000
lr = 0.01
cost = 0
for i in range(n_iter):
    
    w2,w1,b = fit(X1, X2, Y, lr, w1, w2, b, cost)
    
print("w1 : {}, w2 : {}, b : {}".format(w1, w2, b))