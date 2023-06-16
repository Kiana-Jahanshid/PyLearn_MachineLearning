import os
import cv2
import numpy as np
 
    
for i in range(10):
    directory = f"{i}"
    parent_dir = "mnist"
    path = os.path.join(parent_dir, directory) 
    os.makedirs(path)


image = cv2.imread("mnist.webp")
y=0
for k in range(10):

    for i in range(5):
        x=0
        for j in range(100):
            cropped= image[0+y:20+y , 0+x:20+x]
            x+= 20
            img = cv2.imwrite(f"mnist/{k}/{k}{i}{j}.jpg" , cropped)
        y += 20

