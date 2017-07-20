import numpy as np
from brian2 import *
from time import time
from matplotlib import pyplot as plt
from common import read_img, recon_weights, read_dataset
import parameters as par

x = read_img()

normalized = (x-np.min(x))/float(np.max(x)-np.min(x))*50

taupre = 5*ms
taupost = 8*ms
wmax = 1.2
wmin = -0.5
Apre = 0
Apost = -0.4*0.0625

tau = 10*ms
vr = 0*volt
vt = 1.5*mV	
eqs = '''
dv/dt = -v/tau : volt
'''
S_initial = np.random.uniform(low=0,high=0.4,size=(par.num_classes,par.vis_size))

for i in range(1):

      for j in range(1):
            print j

            G = NeuronGroup(1, eqs, threshold='v>vt', reset='v=vr', method='linear')

            # equations governing the dynamics of neurons

            P = PoissonGroup(par.vis_size, x*Hz)

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
            S.w = S_initial
            # M = StateMonitor(G, 'v', record=[0,1,2,3])
            # spikemon = SpikeMonitor(P)
            # spikemon1 = SpikeMonitor(G)
            run(20*ms)
            S_initial = S.w

recon_weights(S_initial)      