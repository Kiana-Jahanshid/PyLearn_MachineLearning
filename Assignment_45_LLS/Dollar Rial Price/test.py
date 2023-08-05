import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 


data = pd.read_csv("Dollar Rial Price\Dollar_Rial_Price_Dataset.csv")
data.drop("Unnamed: 0" , axis =1 , inplace=True)
data["Persian_Date"] = data["Persian_Date"].astype(str)
data["Persian_Date"] = data["Persian_Date"].str.replace("/" , "")

for i in range(len(data["Persian_Date"])) :
    temp = data["Persian_Date"][i]
    data["Persian_Date"][i] = str(temp[0:4]) + "-" + str(temp[4:6])+ "-" + str(temp[6:8])





data["Year"] = [data["Persian_Date"][i][0:4] for i in range(len(data)) ]
data["Year"] = data["Year"].astype(int)
data["presidency"] = None

for i in range(len(data)) :

    if 1384 <= data["Year"][i] and data["Year"][i] <= 1390 :
        data["presidency"][i] = "AHMADI-NEJAD"
    
    elif 1392 < data["Year"][i] and   data["Year"][i] < 1400 :
        data["presidency"][i] = "ROUHAANI"
    
    elif data["Year"][i] >= 1400 :
        data["presidency"][i] = "RAEESI"

#x= data.query("presidency == 'RAEESI' ")
print(data)