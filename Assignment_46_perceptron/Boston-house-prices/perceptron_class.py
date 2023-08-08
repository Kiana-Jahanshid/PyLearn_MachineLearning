import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, lr_w , lr_b , epochs):
        self.w = None
        self.bias = None
        self.lr_w = lr_w
        self.lr_b= lr_b
        self.epochs = epochs


    def fit(self, X_train, Y_train):
        n_samples, n_features = X_train.shape
        print(n_features)
        self.W = np.random.rand(n_features)
        self.Bias = np.random.rand(1 , 1)
        fig , (ax1 , ax2) = plt.subplots(2,1 ,constrained_layout=True)
        losses = []
        for j in range(self.epochs):
            for i in range(len(X_train)):
                x1 = X_train[i,0]
                x2 = X_train[i,1]
                y = Y_train[i]
                print(x1)
                print(x2)
                print(y)
                print(self.W)
                print(self.W[0])

                y_pred = x1 * self.W[0] + x2 * self.W[1]
                error = y - y_pred

                # SGD = it's a update formula (optimization)
                self.W1 = self.W + (error * x1 * self.lr_w)
                self.W2 = self.W + (error * x2 * self.lr_w)
               
                self.Bias = self.Bias + (error * self.lr_b)


                #Y_pred = x1 @ self.W1 + x2 @ self.W2 + self.Bias
                # ax1.clear()
                # ax1.scatter(X_train , Y_train , color="blue" , alpha = 0.6)
                # ax1.plot(X_train , Y_pred , color="red")
                # ax1.set_ylabel("Whole weight ")
                # ax1.set_xlabel("Shucked weight")
                a, b = np.meshgrid(X_train[:,0], X_train[:,1])
                print("------------")
                print(a)
                print(b)
                print(self.W2)
                print(self.W1)
                print(self.Bias)

                plane = self.W1 * a + self.W2 * b + self.Bias 



                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.plot_surface(a, b, plane , alpha= 0.2)
                plt.scatter(X_train[:,0] , X_train[:,1] , Y_train , label="data" )
                plt.legend()
                plt.show()

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