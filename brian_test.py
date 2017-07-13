import numpy as np
from brian2 import *
from matplotlib import pyplot as plt


sigma = 0.2

#equations governing the dynamics of neurons
eqs = '''dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''
#neuron groups and parameters
G = NeuronGroup(3, eqs, threshold = 'v>1', reset = 'v=0', refractory = 5*ms ,method='linear')
G.I = [5, -1,-0]
G.tau = [10, 100,100]*ms

#state monitor to record the membrane voltage
M = StateMonitor(G, 'v', record=[0,1,2])

#synapses
S = Synapses(G, G, on_pre='v_post += 0.2')
S.connect(i=0, j=[1,2])

run(300*ms)

plt.plot(M.t/ms, M.v[0])
plt.plot(M.t/ms, M.v[1])
plt.plot(M.t/ms, M.v[2])
plt.show()