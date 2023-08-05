import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from perceptron_class import Perceptron



data = pd.read_csv("Abalone/abalone.csv")
X = data["Whole weight"].values
Y = data["Shucked weight"].values

X_train , X_test , Y_train , Y_test = train_test_split(X , Y ,shuffle=True ,  test_size=0.4 )
Y_train = Y_train.reshape(-1 , 1)
Y_test = Y_test.reshape( -1, 1)
X_train = X_train.reshape(-1 , 1)
X_test = X_test.reshape( -1, 1)
print(X_test.shape)
print(X_train.shape)
print(Y_train.shape)
print(Y_test.shape)


learning_rate_w = 0.00001
learning_rate_bias = 0.0000001 
Epoch = 40 

perceptron = Perceptron(len(X_train) , learning_rate_w , learning_rate_bias , Epoch) 
perceptron.fit(X_train , Y_train)

y_predicted = perceptron.predict(X_test)
y_predicted_train = perceptron.predict(X_train)
print("-----")
print(y_predicted.shape)
Y_test = Y_test.reshape(Y_test.shape[0])
print(Y_test.shape)
print(y_predicted_train.shape)
Y_train = Y_train.reshape(Y_train.shape[0])
print(Y_train.shape)

