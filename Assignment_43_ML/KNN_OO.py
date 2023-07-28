import numpy as np
import cv2
import matplotlib.pyplot as plt
from knn import KNN
from sklearn.model_selection import train_test_split

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y


    def euclidean_distance(self, x1 , x2):
        return np.sqrt(np.sum((x1 - x2)**2 ))


    def predict(self, list_of_new_datas):
        Y_list = []
        for x in list_of_new_datas :          
            distances = []
            for x_train in self.X_train :
                d = self.euclidean_distance(x , x_train) 
                distances.append(d)
            nearest_neighbour = np.argsort(distances)[0 : self.k]
            result = np.bincount(self.Y_train[nearest_neighbour])
            y = np.argmax(result)
            Y_list.append(y)
        return Y_list     
    
    def evaluate(self, X, Y):
        Y_predicted = self.predict(X)     
        accuracy = np.sum(Y_predicted == Y)  / len(Y)           
        return accuracy





class FindingNemo:
    def __init__(self, train_image):

        self.train_image = train_image
        
    
    def convert_image_to_dataset(self, image):
        global pixels_list_hsv , final_mask
        hsv_img = cv2.cvtColor(image , cv2.COLOR_RGB2HSV)
        pixels_list_hsv = hsv_img.reshape(-1 ,3)
        x_train = pixels_list_hsv / 255 

        light_orange = (1, 190, 200)
        dark_orange = (18, 255, 255)
        mask1 = cv2.inRange(hsv_img, light_orange, dark_orange) 

        light_white = (0, 0, 200)
        dark_white = (145, 60, 255)
        mask2 = cv2.inRange(hsv_img , light_white ,dark_white  )
        final_mask =  mask1 + mask2
        final_result = cv2.bitwise_and(image, image, mask=final_mask)
        y_train = final_mask.reshape(-1 ,) // 255

        return x_train , y_train


    def remove_background(self, test_image):
        global dashe_hsv_img
        test1 = cv2.resize(test_image , (0,0) , fx=0.25 , fy=0.25)
        test2 = cv2.cvtColor(test1 , cv2.COLOR_BGR2RGB)
        dashe_hsv_img = cv2.cvtColor(test2 , cv2.COLOR_RGB2HSV)

        x_test  = dashe_hsv_img.reshape(-1,3) / 255 
        x_test = np.array(x_test)
        return x_test


    def preprocess(self , image):
        image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
        image = cv2.resize(image , (0,0) , fx=0.2 , fy=0.2)
        return image

    def train(self , x_train , y_train):
        global knn
        knn = KNN(3)
        knn.fit(x_train , y_train)

    def test(self , test_image):
        x_test = nemo.remove_background(test_image)
        Y_pred = knn.predict(x_test)
        Y_pred = np.array(Y_pred)
        output1 = Y_pred.reshape(dashe_hsv_img.shape[:2])
        output1 = output1.astype("uint8")
        final_result = cv2.bitwise_and(dashe_nemo , dashe_nemo, mask=output1)
        return final_result

if __name__=="__main__":

    # train :
    train_image = cv2.imread("IBM/pylearn/Assignment_43_ML/nemo.jpg")
    nemo = FindingNemo(train_image)
    init_img = nemo.preprocess(train_image)
    X_train , Y_train = nemo.convert_image_to_dataset(init_img)
    nemo.train(X_train , Y_train)


    #test :
    dashe_nemo = cv2.imread("IBM\pylearn\Assignment_43_ML\dashe-nemo.jpg")
    pred_result = nemo.test(dashe_nemo)
    plt.imshow(pred_result , cmap="gray")
    plt.savefig("dashe__nemo.jpg")


