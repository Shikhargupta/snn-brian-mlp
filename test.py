import numpy as np
import pandas as pd
import cv2
from brian2 import *
from brian_eval import evaluate

# W = []
# for i in range(10):
# 	img = cv2.imread("weights/weights" + str(i)+".png",0)
# 	img = np.ndarray.flatten(img)
# 	W.append(img)

W = pd.read_csv("weights/weights.csv")
W = W.as_matrix()
W = W[:,1]

# df = pd.read_csv("data/net2/test.csv")
# df = df.as_matrix()

df = pd.read_csv("data/net2_inverted/inverted.csv")
df = df.as_matrix()

# # print df
# # print np.shape(df)
x = df[:,1:]
y = df[:,0]

x = (x-np.min(x))/float(np.max(x)-np.min(x))*200

# W = np.asarray(W)
# W = W.astype(float)
# W = W/255

# print W
evaluate(W,x,y)
