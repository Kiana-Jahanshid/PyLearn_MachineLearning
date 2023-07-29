import numpy as np
import math
from numpy.linalg import inv 



class LinearLeastSquare:

    def __init__(self):
        self.w = None 

    def fit(self, x_train , y_train):
        self.w = np.matmul(  np.matmul(np.linalg.inv( np.matmul(x_train.T  , x_train) ) , x_train.T ) , y_train )

    def predict(self , x_test):
        y_pred = x_test @ self.w
        return y_pred

    def evaluate(self , x_test , y_test , metric):
        y_pred = self.predict(x_test)
        error = y_test - y_pred 
        if metric == "mae" :
            loss = np.sum(np.abs(y_test - y_pred)) / len(y_test)
        elif metric == "mse":
            loss = np.sum((y_test - y_pred) ** 2) / len(y_test)
        elif metric == "rmse" :
            loss = math.sqrt(np.sum((y_test - y_pred) ** 2) / len(y_test))

        return loss 