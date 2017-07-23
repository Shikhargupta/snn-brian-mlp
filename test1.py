import numpy as np
import pandas as pd
import cv2


# W = []
# for i in range(10):
# 	img = cv2.imread("test/" + str(i+1)+".png",0)
# 	img = np.ndarray.flatten(img)
# 	W.append(img)

# df = pd.DataFrame(W)
# df.to_csv("test_data.csv")

df = pd.read_csv("data/net2_inverted/inverted.csv")
df = df.as_matrix()
d0 = []
d1 = [] 
d2 = []
d3 = []
d4 = []
d5 = []
d6 = []
d7 = []
d8 = []
d9 = []

for i in range(np.shape(df)[0]):
	if df[i,0]==0:
		d0.append(df[i,1:])
	elif df[i,0]==1:
		d1.append(df[i,1:])
	elif df[i,0]==2:
		d2.append(df[i,1:])
	elif df[i,0]==3:
		d3.append(df[i,1:])
	elif df[i,0]==4:
		d4.append(df[i,1:])
	elif df[i,0]==5:
		d5.append(df[i,1:])
	elif df[i,0]==6:
		d6.append(df[i,1:])
	elif df[i,0]==7:
		d7.append(df[i,1:])
	elif df[i,0]==8:
		d8.append(df[i,1:])
	elif df[i,0]==9:
		d9.append(df[i,1:])									

df_0 = pd.DataFrame(d0)
df_0.to_csv("data/net2/0.csv")

df_1 = pd.DataFrame(d1)
df_1.to_csv("data/net2/1.csv")

df_2 = pd.DataFrame(d2)
df_2.to_csv("data/net2/2.csv")

df_3 = pd.DataFrame(d3)
df_3.to_csv("data/net2/3.csv")

df_4 = pd.DataFrame(d4)
df_4.to_csv("data/net2/4.csv")

df_5 = pd.DataFrame(d5)
df_5.to_csv("data/net2/5.csv")

df_6 = pd.DataFrame(d6)
df_6.to_csv("data/net2/6.csv")

df_7 = pd.DataFrame(d7)
df_7.to_csv("data/net2/7.csv")

df_8 = pd.DataFrame(d8)
df_8.to_csv("data/net2/8.csv")

df_9 = pd.DataFrame(d9)
df_9.to_csv("data/net2/9.csv")