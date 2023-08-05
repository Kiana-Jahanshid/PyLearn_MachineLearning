import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from perceptron_class import Perceptron
from sklearn import datasets




x, y, coef = datasets.make_regression(n_samples=300, n_features=1 ,n_informative=1,noise=10,coef=True,random_state=40) 

# Scale feature x (years of experience) to range 0..20
X = np.interp(x, (x.min(), x.max()), (0, 20))
# Scale target y (salary) to range 20000..150000 
Y = np.interp(y, (y.min(), y.max()), (20000, 150000))


X_train , X_test , Y_train , Y_test = train_test_split(X , Y ,shuffle=True ,  test_size=0.2)
Y_train = Y_train.reshape(-1 , 1)
Y_test = Y_test.reshape( -1, 1)
print(X_test.shape)
print(X_train.shape)
print(Y_train.shape)
print(Y_test.shape)



learning_rate_w = 0.00001  
learning_rate_bias = 0.0001 
Epoch = 10 
Y_test_ = []
X_test_ = []

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
