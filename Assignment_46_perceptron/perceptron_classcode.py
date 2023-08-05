import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
import time





data = pd.read_csv("Assignment_46_perceptron\weight-height.csv")

X = data["Height"].values
Y = data["Weight"].values
X_train , X_test , Y_train , Y_test = train_test_split(X , Y ,shuffle=True ,  test_size=0.3)

print(X_train.shape)
X_train = X_train.reshape(-1 , 1)
print(X_train.shape)
X_test = X_test.reshape(-1 , 1)
Y_train = Y_train.reshape(-1 , 1)
Y_test = Y_test.reshape(-1 , 1)




W = np.random.rand(1,1)
Bias  = np.random.rand(1,1)


fig , (ax1 , ax2) = plt.subplots(2,1)
learning_rate_w = 0.0001 # ta kond update beshe 
learning_rate_bias = 0.01 # ta sari update beshe 
Epoch = 40 # 50 bar data ha ro 
losses = []


for j in range(Epoch):
    for i in range(X_train.shape[0]):
        x = X_train[i]
        y = Y_train[i]

        y_pred = x @ W
        error = y - y_pred

        # SGD = it's a update formula (optimization)
        W = W + (error * x * learning_rate_w)
        Bias = Bias + (error * learning_rate_bias)

        #print(W)
        #time.sleep(0.5)

        Y_pred = X_train * W + Bias
        ax1.clear()
        ax1.scatter(X_train , Y_train , color="blue")
        ax1.plot(X_train , Y_pred , color="red")


        # MAE loss 
        loss = np.mean(np.abs(error))
        losses.append(loss)

        # MSE loss 
        #loss = np.mean(np.abs(error**2))
        #losses.append(loss)


        ax2.clear()
        ax2.plot(losses)
        plt.pause(0.01)
