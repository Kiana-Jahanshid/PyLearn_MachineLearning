import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, input_size, lr_w , lr_b , epochs):
        self.w = None
        self.bias = None
        self.input_size = input_size
        self.lr_w = lr_w
        self.lr_b= lr_b
        self.epochs = epochs


    def fit(self, X_train, Y_train):
        #n_samples, n_features = X_train.shape

        self.W = np.random.rand(1 , 1)
        self.Bias = np.random.rand(1 , 1)
        fig , (ax1 , ax2) = plt.subplots(2,1 ,constrained_layout=True)
        losses = []
        for j in range(self.epochs):
            for i in range(X_train.shape[0]):
                x = X_train[i]
                y = Y_train[i]

                y_pred = x @ self.W
                error = y - y_pred

                # SGD = it's a update formula (optimization)
                self.W = self.W + (error * x * self.lr_w)
                self.Bias = self.Bias + (error * self.lr_b)


                Y_pred = X_train * self.W + self.Bias
                ax1.clear()
                ax1.scatter(X_train , Y_train , color="blue" , alpha = 0.6)
                ax1.plot(X_train , Y_pred , color="red")
                ax1.set_ylabel("Whole weight ")
                ax1.set_xlabel("Shucked weight")

            # MAE loss 
            #loss = np.mean(np.abs(error))
            #losses.append(loss)

            # MSE loss 
            loss = np.mean(np.abs(error**2))
            losses.append(loss)
            ax2.clear()
            ax2.plot(losses)
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("LOSS")
            plt.pause(0.01)

        plt.savefig("111_result.jpg")

    def predict(self, X_test):
        y_pred = X_test @ self.W + self.Bias
        return y_pred