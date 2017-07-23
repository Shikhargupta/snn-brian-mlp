import pandas as pd
import numpy as np
import cv2


df = pd.read_csv("4.csv")
df = df.as_matrix()

a = df[5,1:]
summ = np.zeros((1,100))
for i in range(500):
	summ = summ + df[i,1:]
summ = summ/500	
a = np.resize(summ,(10,10))*255
cv2.imwrite("img.png",a)