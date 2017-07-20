
############################################# README ###########################################################################

# This is a demo showing how to train a spiking neural network for 2 very distinct patterns using STDP learning rule. Neuron populations
# are defined for input and output layers and connected using synapses which are updated using stdp rule.

################################################################################################################################ 

from common import *

x,y = read_dataset()

#normalizing data
normalized = (x-np.min(x))/float(np.max(x)-np.min(x))*50 #50 Hz is the maximum firing rate

#stdp parameters
sigma = 0.0625
taupre = 5*ms
taupost = 8*ms
wmax = 1.2
wmin = -0.5
Apre = 0.4*sigma
Apost = -0.2*sigma

#characteristics of neurons
tau = 10*ms
vr = 0*volt
vt = 1.5*mV

# equations governing the dynamics of neurons
eqs = '''
dv/dt = -v/tau : volt (unless refractory)
'''
# initializing weights
S_initial = np.random.uniform(low=0,high=0.4,size=(par.hid_size,par.vis_size))

for i in range(par.epochs):

      for j in range(len(x)):

            #output layer neuron
            G = NeuronGroup(1, eqs, threshold='v>vt', reset='v=vr', refractory = 1.5*ms, method='linear')

            #input neurons firing according to Poisson distribution with rates determined by the intensity of the corresponding input pixel
            P = PoissonGroup(par.vis_size, x[j]*Hz)

            #synapse governed by the rules of STDP
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
            S.w = S_initial[j]

            M = StateMonitor(G, 'v', record=True) #monitors the membrane voltage
            spikemon = SpikeMonitor(P) #records spikes from input neurons
            spikemon1 = SpikeMonitor(G) #records spikes from output neurons
            run(200*ms)

            #updating the weights
            S_initial[j] = S.w

######################################################### plots ##################################################
            # subplot(211)
            # plot(spikemon.t/ms, spikemon.i, '.k')
            # subplot(212)
            # plot(spikemon1.t/ms, spikemon1.i, '.k')
            # plot(M.t/ms, M.v[0]/volt)
            # show()
###################################################################################################################

#reconstructing weights to analyse the training
recon_weights(S_initial)