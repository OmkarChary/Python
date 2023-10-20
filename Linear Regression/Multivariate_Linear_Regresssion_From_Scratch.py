import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    
    def __init__(self, n_iter = 1000, alpha = 0.001 ):
        self.n_iter = n_iter
        self.alpha = alpha
        self.weights = None
        
    def fit( self, X, y ):
        
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features + 1)
        
        X = X.append(np.ones(n_sample, 1))
        
        for _ in range(self.n_iter):
            linear_model = np.dot()
      
        
        
        
        
        


df = pd.read_csv('/Users/omkarchary/Documents/Python/Linear Regression/Real-estate1.csv')



X = pd.drop(df['Y house price of unit area'], axis = 1)
y = df['Y house price of unit area']