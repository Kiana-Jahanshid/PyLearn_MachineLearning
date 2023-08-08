import numpy as np
import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(8,7))
ax1 = fig.add_subplot(211, projection="3d")
ax2 = fig.add_subplot(212)


class Perceptron:
    def __init__(self, X_test, Y_test, lr_w , lr_b , epochs):
        self.w = None
        self.bias = None
        self.lr_w = lr_w
        self.lr_b= lr_b
        self.epochs = epochs
        self.X_test = X_test
        self.Y_test = Y_test

    def fit(self, X_train, Y_train):
        n_features = X_train.shape[1]
        self.W = np.random.rand(n_features)
        self.Bias = np.random.rand(1 , 1)

        losses = []
        for j in range(self.epochs):
            for i in range(X_train.shape[0]):
                X = X_train[i]
                y = Y_train[i]
                y_pred = X @ self.W
                error = y - y_pred

                # SGD = it's a update formula (optimization)
                self.W = self.W + (error * X * self.lr_w)
                self.Bias = self.Bias + (error * self.lr_b)

                Y_pred = self.X_test * self.W + self.Bias
                ERROR =   self.Y_test -  Y_pred


            a, b = np.meshgrid(X_train[:,0], X_train[:,1])
            plane = self.W[0] *  a + self.W[1] * b 
            ax1.clear()
            ax1.scatter(X_train[:,0] , X_train[:,1] ,  Y_train , color="green" , alpha = 0.6)
            ax1.plot_surface(a, b, plane , alpha= 0.2)
            ax1.set_ylabel("TAX")
            ax1.set_xlabel("NOX")
            ax1.set_zlabel("PRICE")


            # MAE loss 
            #loss = np.mean(np.abs(ERROR))
            #losses.append(loss)

            # MSE loss
            loss = np.mean(np.abs(ERROR**2))
            losses.append(loss)

            ax2.clear()
            ax2.plot(losses)
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("LOSS")
            ax2.set_title("BOSTON HOUSE PRICE")
            plt.pause(2)


    def predict(self, X_test):
        self.y_pred = X_test @ self.W + self.Bias
        return self.y_pred

    def eval(self , Y_test):
        ERROR = Y_test - self.y_pred
        loss = np.mean(np.abs(ERROR**2))
        return loss
