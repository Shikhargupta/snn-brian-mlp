import numpy as np
from brian2 import *
from matplotlib import pyplot as plt


sigma = 0.2

eqs = '''dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''

G = NeuronGroup(2, eqs, threshold = 'v>1', reset = 'v=0', refractory = 5*ms ,method='linear')
G.I = [5, -1]
G.tau = [10, 100]*ms
M = StateMonitor(G, 'v', record=[0,1])

S = Synapses(G, G, on_pre='v_post += 0.2')
S.connect(i=0, j=1)

run(300*ms)

plt.plot(M.t/ms, M.v[1])
plt.show()