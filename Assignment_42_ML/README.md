# KNN using ANSUR II dataset :

# 2 . Preprocess stage :

### convert weight to kg 
> concatenated_dataframes['weightkg'] =  concatenated_dataframes['weightkg'] / 10

### calc heigth , convert it to cm --- > using (((  stature  )))  column 

> concatenated_dataframes['stature'] = concatenated_dataframes['stature'] / 10


### convert Gender column (str) ----> to number 
### female = 0 
### male   = 1

> concatenated_dataframes['Gender'] = concatenated_dataframes['Gender'].replace(["Female" , "Male"] , [ 0 , 1 ])

#
#
#
# 3. Show heights for women and men on same plot.

> < img src=(https://github.com/kiana-jahanshid/PyLearn_MachineLearning/blob/main/Assignment_42_ML/outputs/women%20and%20men%20height.png)  width="500">

## A. Why is the data of men higher than the data of women?

> 🟡 according to Dataset explanation  , the number of each Gender is (4,082 men and 1,986 women) , therefore the height of bins for men are taller than women . 

## ‌B. Why is the data of men more right than the data of women?
> 🟠 because the mean of men's height is greater than mean of women's height . 

## ✅ Answer :  

### 🚹 Mean of Women height =  162.84 cm 
### 🚺 Mean of Men height   =  175.62 cm 

#
#
#
#

# 4. Split dataset to train and test datasets (%80 for train and %20 for test):

> X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=0.2, random_state=300)

> ## len(X_train) =  6068 * 0.8 = 4854 
> ## len(X_test) = 6068 * 0.2 = 1214

#
#
#

# 5. fit your OOP-KNN algorithm on the train dataset.

+ + + + + + + + 


|  |  K = 3  |  K = 5  |  K = 7 |
|---------------| --------------- | --------------- | --------------- |
|2 Features ("stature" , "weightkg")|  0.83   | 0.84  | 0.84  |
|3 Features ("stature" , "weightkg" , "biacromialbreadth") |  0.91   | 0.916  | 0.917  |
|4 Features (stature" , "weightkg" , "biacromialbreadth" , "shouldercircumference")|  0.950  | 0.953  | 0.953  |


+ + + + + + + +  
# 
# 10. Calculate confusion matrix using scikit-learn:

< img src=(https://github.com/kiana-jahanshid/PyLearn_MachineLearning/blob/main/Assignment_42_ML/outputs/conf_matrix.png) width="500" >
