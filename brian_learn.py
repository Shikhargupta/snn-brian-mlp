import numpy as np
from brian2 import *
from time import time
from matplotlib import pyplot as plt
from common import read_img, recon_weights, read_dataset
import parameters as par
from brian_eval import evaluate
import pandas as pd


# x,y, x_test, y_test = read_dataset()
x = read_dataset()


# x_test = (x_test-np.min(x_test))/float(np.max(x_test)-np.min(x_test))
normalized = (x-np.min(x))/float(np.max(x)-np.min(x))

# x_test[x_test<0.5] = 0
# x_test[x_test>0.5] = 0.9
normalized[normalized<0.5] = 0
normalized[normalized>=0.5] = 0.9
# x_test = x_test*50
normalized = normalized*50

sigma = 0.0625
taupre = 5*ms
taupost = 5*ms
wmax = 1
wmin = -1
Apre = 0.8*sigma
Apost = -0.2*sigma

tau = 10*ms
vr = 0*mV
vt = 1.5*mV	
eqs = '''
dv/dt = -v/tau : volt (unless refractory)
'''
S_initial = np.random.uniform(low=0,high=0.4,size=(par.num_classes*par.neurons_per_class,par.vis_size))

for i in range(par.epochs):

      for j in range(np.shape(normalized)[0]):
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
            for ind in range(par.vis_size):
                  if normalized[j][ind]==0:
                        S.w[ind] -= 0.008

            # S.w[normalized[j]==0] -= 20
            S_initial = S.w

            if (j+1)%par.test_every==0:
                  recon_weights(S_initial)
                  df = pd.DataFrame(np.asarray(S_initial))
                  df.to_csv("weights/weights.csv")
                  # print "processed " + str(j+1) + " samples."
                  # acc = evaluate(S_initial ,x_test,y_test)

recon_weights(S_initial)