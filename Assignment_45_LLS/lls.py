import numpy as np
from numpy.linalg import inv 



class LinearLeastSquare:

    def __init__(self):
        self.w = None 

    def fit(self, x_train , y_train):
        self.w = inv(x_train.T @ x_train) @ x_train.T @  y_train
        return self.w


    def predict(self , x_test):
        y_pred = x_test @ self.w
        return y_pred

    def evaluate(self , x_test , y_test , metric="mse"):

        y_pred = self.predict(x_test)
        error = y_test - y_pred 
        if metric == "mae" :
            loss = np.sum(np.abs(y_test - y_pred)) / len(y_test)
        elif metric == "mse":
            loss = np.sum(np.abs(y_test - y_pred)) / len(y_test)

        return loss 