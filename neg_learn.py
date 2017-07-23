import numpy as np
from brian2 import *
from time import time
from matplotlib import pyplot as plt
from common import read_img, recon_weights, read_dataset
import parameters as par
from brian_eval import evaluate
import pandas as pd

for cl in range(10):
	if cl!=2:

		df = pd.read_csv("data/net2_inverted/8.csv")
		df = df.as_matrix()
		x = df[:par.num_train,1:]

		normalized = (x-np.min(x))/float(np.max(x)-np.min(x))
		normalized[normalized<0.5] = 0
		normalized[normalized>=0.5] = 0.9
		# x_test = x_test*50
		normalized = normalized*50

		sigma = 0.0625
		taupre = 5*ms
		taupost = 5*ms
		wmax = 1
		wmin = -1
		Apre = -0.5*sigma
		Apost = 0*sigma

		tau = 10*ms
		vr = 0*mV
		vt = 1.5*mV	
		eqs = '''
		dv/dt = -v/tau : volt (unless refractory)
		'''
		S_initial = pd.read_csv("weights/weights.csv")
		S_initial = S_initial.as_matrix()
		S_initial = S_initial[:,1]

		for i in range(par.epochs):

		      for j in range(100):
		            print j

		            G = NeuronGroup(1, eqs, threshold='v>vt', reset='v=vr', refractory = 1.5*ms, method='linear')

		            # equations governing the dynamics of neurons

		            P = PoissonGroup(par.vis_size, normalized[j]*Hz)

		            S = Synapses(P, G,
		                         '''
		                         w : 1
		                         dapre/dt = -apre/taupre : 1 (event-driven)
		                         dapost/dt = -apost/taupost : 1 (event-driven)
		                         ''',
		                         on_pre='''
		                         v_post += w*mV
		                         apre += Apre
		                         w = clip(w+apost, wmin, wmax)
		                         ''',
		                         on_post='''
		                         apost += Apost
		                         w = clip(w+apre, wmin, wmax)
		                         ''')

		            S.connect()
		            # print np.shape(S_initial[y[j]])
		            # S.w = S_initial[y[j]]
		            S.w = S_initial
		            # M = StateMonitor(G, 'v', record=[0,1,2,3])
		            # spikemon = SpikeMonitor(P)
		            spikemon1 = SpikeMonitor(G)
		            run(30*ms)

		            # S.w[normalized[j]==0] -= 20
		            S_initial = S.w

		recon_weights(S_initial)