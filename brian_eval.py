import numpy as np
from brian2 import *
from time import time
from matplotlib import pyplot as plt
from common import read_img, recon_weights, read_dataset
import parameters as par
import pandas as pd

def evaluate(W,x,y):

	tau = 20*ms
	vr = 0*mV
	vt = 1000*mV	
	eqs = '''
	dv/dt = -v/tau : volt (unless refractory)
	'''
	acc = 0
	for j in range(len(x)):

		# print "sample no " + str(j)

		# G = NeuronGroup(par.hid_size, eqs, threshold='v>vt', reset='v=vr', refractory = 1.5*ms, method='linear')
		# # print x
		# P = PoissonGroup(par.vis_size, x*Hz) #flag!!

		# S = Synapses(P, G, 'w : 1', on_pre='v_post += w*volt')
		# S.connect()

		# S.w = np.reshape(np.ndarray.flatten(W), (1,7840))
		# # temp = S.w[:784]
		# # print np.shape(temp)
		# # recon_weights(temp)
		# # S.w = np.reshape(S.w, (1,7840))
		# # print np.shape(S.w)
		# # for i in range(np.shape(S.w)[0]):
		# # 	if S.w[i]==0:
		# # 		S.w[i] = -0.02

		# statemon = StateMonitor(G, 'v', record=True)
		# M = SpikeMonitor(G)
		# run(300*ms)
		# # plot(statemon.t/ms, statemon.v[0]/volt)
		# # show()
		# spikes = np.zeros((par.hid_size,1))
		# # print M.i
		# for s in M.i:
		# 	# print s
		# 	spikes[s] += 1

		# eval_class = np.argmax(spikes)
		# print spikes
		# print eval_class
		# for nu in range(1):
		# 	plot(statemon.t/ms, statemon.v[nu]/volt)
		# show()	
		
		

		# true_class = y[j]

		# if int(eval_class)==int(true_class):
		# 	acc += 1

	# accuracy = float(acc)/par.num_test
	# print "accuracy: " + str(accuracy)

		G = NeuronGroup(1, eqs, threshold='v>vt', reset='v=vr', refractory = 1.5*ms, method='linear')
		P = PoissonGroup(par.vis_size, x[j]*Hz) #flag!!
		S = Synapses(P, G, 'w : 1', on_pre='v_post += w*volt')
		S.connect()

		S.w = W
		spikemon = SpikeMonitor(G)
		run(30*ms)
		print "for sample " + str(y[j]) + " " + str(len(spikemon))
		# activity.append(len(spikemon))
	# eval_class = np.argmax(activity)
	# true_class = y[j]
	# if int(eval_class)==int(true_class):
	# 	acc += 1

# accuracy = float(acc)/par.num_test
# print "accuracy is " + str(accuracy)