import numpy as np
import pandas as pd
import cv2
from numpy import interp
import parameters as par

def read_img():


	img = cv2.imread("101.png", 0)
	img2 = cv2.imread("101.png",0)

	img = np.ndarray.flatten(img)
	img2 = np.ndarray.flatten(img2)

	data_mat = [img]
	df = pd.DataFrame(data_mat)
	df.to_csv("data.csv")
	return img

def recon_weights(W):

	for i in range(par.hid_size):
		temp = W
		recon = np.reshape(temp,(par.x_pixel,par.x_pixel))*255
		cv2.imwrite("weights/weights" + str(i) + ".png",recon)

def read_dataset():
	df = pd.read_csv("data/net2_inverted/2.csv")
	df = df.as_matrix()
	X = df[:par.num_train,1:]
	# y = df[:par.num_train,0]
	# x_test = df[par.num_train:par.num_train+par.num_test,1:]
	# y_test = df[:par.num_train+par.num_test,0]
	# return X,y,x_test,y_test
	return X