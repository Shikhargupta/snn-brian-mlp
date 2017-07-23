import pandas as pd
import numpy as np
import cv2

df = pd.read_csv("../net2.csv")
df = df.as_matrix()

x = df[:,1:]
y = df[:,0]

x = (x-np.min(x))/float(np.max(x)-np.min(x))

x[:] = [abs(1-a) for a in x]
y = np.reshape(y,(42000,1))
print np.shape(y)
inv = np.hstack((y,x))

df1 = pd.DataFrame(inv)
df1.to_csv("inverted.csv")