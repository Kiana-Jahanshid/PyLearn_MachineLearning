import numpy as np 


class KNN :
    def __init__(self , k ) :
        self.k = k 


    # KNN training stage 
    def fit(self , X , Y): 
        self.X_train = X
        self.Y_train = Y


    # predict & GIVE NEW FRUIT label (y) ---> apple or banana
    # gets fruits FEATURE
    def predict(self , list_of_new_datas): 
        Y_list = []
        for x in list_of_new_datas :
            
            distances = []
            for x_train in self.X_train :
                d = self.euclidean_distance(x , x_train) # euclidean_distance between new data & each of the TRAIN DATA
                distances.append(d)

            nearest_neighbour = np.argsort(distances)[0 : self.k]
            result = np.bincount(self.Y_train[nearest_neighbour])
            y = np.argmax(result)
            Y_list.append(y)

            
        # we must return a list of y's ,, not only a one y 
        return Y_list 



    def euclidean_distance(self , x1 , x2) :
        return np.sqrt(np.sum((x1 - x2)**2 ))
    


    # khode in tabe javabo midoone (x & y az ghabl moshakhas hast), va mikhad az ma beporse va test begire
    # x == questions 
    # y == answers
    # goal = compareing  our answers ,, with their true results 
    def evaluate(self , X , Y ) : 

        Y_predicted = self.predict(X)     # our answers     # COMPARE ((( Y == real answer))) and  (((( Y_predicted == our prediction)))
        accuracy = np.sum(Y_predicted == Y)  / len(Y)               # CHECK "how many" of (((Y_predicted))) are equal to  (((( Y ))))
        return accuracy