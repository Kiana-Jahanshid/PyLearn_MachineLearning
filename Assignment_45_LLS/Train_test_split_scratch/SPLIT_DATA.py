import numpy  as np 

class TrainTestSplit_FromSCRATCH :
    def __init__(self , test_ratio , shuffle_data):
        self.test_ratio = test_ratio
        self.shuffle_data = shuffle_data


    def shuffle(self  ,X, Y):
        global x_datas_indices
        x_datas_indices = np.arange(len(X))
        np.random.shuffle(x_datas_indices)
        shuffled_X = X[x_datas_indices]
        shuffled_Y = Y[x_datas_indices]
        return shuffled_X , shuffled_Y
    

    def split(self , X , Y) :

        if self.shuffle_data == True :
            np.random.seed(40) 
            test_size = int(len(X) * self.test_ratio)
            X_shuff  , Y_shuff= self.shuffle(X , Y)
            train_size = len(Y) - test_size
            X_train, X_test = X_shuff[0:train_size], X_shuff[train_size:]
            y_train, y_test = Y_shuff[0:train_size], Y_shuff[train_size:]
            return X_train, X_test, y_train, y_test

        elif self.shuffle_data == False:
            np.random.seed(40) 
            test_size = int(len(X) * self.test_ratio)
            TRAIN_INDICES = x_datas_indices[0:test_size]
            TEST_INDICES = x_datas_indices[test_size:]
            X_train, y_train = X[TRAIN_INDICES], Y[TRAIN_INDICES]
            X_test, y_test = X[TEST_INDICES], Y[TEST_INDICES]

            return X_train , X_test , y_train , y_test
        