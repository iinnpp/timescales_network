# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:52:06 2023

@author: mchini

modified version of LIF model
based on Tom's model, based on Wang's model
simplified, with more randomness and a different input
"""

import numpy as np
from scipy.io import savemat
import pickle
from brian2 import *
import networkx as nx

rng = np.random.default_rng()


def get_spike_matrix(spike_monitor, num_neurons, len_stim):
    # initialize
    spike_matrix = zeros((num_neurons, len_stim + 1), dtype=bool)
    # loop over neurons that fired at least once
    for neuron_idx in unique(spike_monitor.i):
        # extract spike_times (in seconds)
        spike_times = spike_monitor.t[spike_monitor.i == neuron_idx]
        # convert them to milliseconds
        spike_times = round_(asarray(spike_times) * 1000).astype(int)
        spike_matrix[neuron_idx, spike_times] = 1
    return spike_matrix


def get_degrees(nPre, nPost, connectivity, lognormal_syn_number):
    # make lognormal distribution between 0 and 1
    # I lowered the SD so that I have fewer issues with neurons having connectivity values exceeding nPost
    min_deg = -1
    while min_deg < 0:
        if lognormal_syn_number > 0.5:
            degrees = lognormal(0, 0.5, size=nPre)
        else:
            degrees = normal(mean(lognormal(0, 0.5, size=nPre)),
                             mean(lognormal(0, 0.5, size=nPre)) / 4,
                             size=nPre)
        min_deg = min(degrees)
    degrees = degrees / degrees.max()
    # convert the lognormal dist to one with sum = nPre * nPost * connectivity
    degrees = np.round(degrees / degrees.sum() * nPre * nPost * connectivity)
    # replace values that are too large (> nPost)
    if degrees.max() > nPost - 1:
        to_replace = lognormal(size=count_nonzero(degrees > nPost - 1))
        to_replace = ceil(to_replace / to_replace.max() * (nPost - 1))
        degrees[degrees > nPost - 1] = to_replace
    return degrees


def get_bin_mat(nPre, nPost, connectivity, norm_deg_pre, norm_deg_post, lognormal_syn_number, prop_syn_number):
    try:
        # if there are no prior constraints on out-degree
        if any(isnan(norm_deg_pre)):
            # make a new out-degree distribution
            out_deg = get_degrees(nPre, nPost, connectivity, lognormal_syn_number)
            # convert number of out-going connections to a probability dist. (sum = 1)
            norm_deg_pre = out_deg / sum(out_deg)
        else:  # if there are constraints on out-degree
            # make number of outgoing connections proportional to norm_deg_pre   
            sources = rng.choice(nPre, int(nPre * nPost * connectivity), p=norm_deg_pre)
            out_deg = histogram(sources, bins=int(max(sources)) + 1,
                                range=(0, max(sources) + 1))[0]
        # if there are no prior constraints on in-degree
        if any(isnan(norm_deg_post)):
            # if nPre is not equal to nPost, nPost is from a different population so there are no constraints
            # if prop_syn_number, then we do not want constraints either
            if nPre != nPost or prop_syn_number < 0.5:
                in_deg = get_degrees(nPost, nPre, connectivity, lognormal_syn_number)
            else:  # if nPre == nPost, the two populations are the same (e.g. E to E connections)
                # make number of incoming connections proportional to norm_deg_pre (same population!)   
                targets = rng.choice(nPost, int(nPre * nPost * connectivity), p=norm_deg_pre)
                in_deg = histogram(targets, bins=int(max(targets)) + 1,
                                   range=(0, max(targets) + 1))[0]
        else:  # if there are constraints on in-degree
            # make number of incmoing connections proportional to norm_deg_post
            targets = rng.choice(nPost, int(nPre * nPost * connectivity), p=norm_deg_post)
            in_deg = histogram(targets, bins=int(max(targets)) + 1,
                               range=(0, max(targets) + 1))[0]
        # now make sure that in_deg.sum() == out_deg.sum() or correct for it
        deg_diff = out_deg.sum() - in_deg.sum()
        if nPre < nPost:
            in_deg[randint(0, len(in_deg))] += deg_diff
        else:
            out_deg[randint(0, len(out_deg))] += - deg_diff
        bin_mat = nx.to_numpy_matrix(nx.directed_havel_hakimi_graph(in_deg, out_deg.astype(int)))
    except nx.NetworkXError:
        bin_mat = NaN
    return bin_mat


########### saving and plotting stuff ###########
root_dir = 'D://snn_from_mattia//output//network_composition//'  # TO CHANGE
v = '_rand'

########### network parameters ###########
n_neurons = 400
# PYRsProp = 0.8
# nPYRS = int(n_neurons * PYRsProp)
# nINs = int(n_neurons - nPYRS)
defaultclock.dt = 0.1 * ms
voltage_clock = Clock(dt=5 * ms)  # use different clock to change sampling rate
simulation_time = 10 * second
PYRs2keep = 320
INs2keep = 80
n_reps = 5000  # number of times to repeat simulation

# Neuron model
CmE = 0.5 * nF  # Membrane capacitance of excitatory neurons
CmI = 0.2 * nF  # Membrane capacitance of inhibitory neurons
gLeakE = 25.0 * nS  # Leak conductance of excitatory neurons
gLeakI = 20.0 * nS  # Leak conductance of inhibitory neurons
Vl = -70.0 * mV  # Leak membrane potential
Vthr = -52.0 * mV  # Spiking threshold
Vrest = -59.0 * mV  # Reset potential
refractoryE = 2 * ms  # refractory period
refractoryI = 1 * ms  # refractory period
TauE = 20 * ms
TauI = 10 * ms

# Synapse model
VrevE = 0 * mV  # Reversal potential of excitatory synapses
VrevI = -80 * mV  # Reversal potential of inhibitory synapses
tau_AMPA_E = 2.0 * ms  # Decay constant of AMPA-type conductances
tau_AMPA_I = 1.0 * ms  # Decay constant of AMPA-type conductances
tau_GABA = 5.0 * ms  # Decay constant of GABA-type conductances

########### excitatory input parameters ###########
num_inputs = 100

# Neuron equations
eqsPYR = '''
dV/dt = (-gea*(V-VrevE) - gi*(V-VrevI) - (V-Vl)) / (tau) + sigma/tau**.5*xi : volt
dgea/dt = -gea/(tau_AMPA_E) : 1
dgi/dt = -gi/(tau_GABA) : 1
tau : second
Cm : farad
sigma : volt
'''
eqsIN = '''
dV/dt = (-gea*(V-VrevE) - gi*(V-VrevI) - (V-Vl)) / (tau) + sigma/tau**.5*xi: volt
dgea/dt = -gea/(tau_AMPA_I) : 1
dgi/dt = -gi/(tau_GABA) : 1
tau : second
Cm : farad
sigma : volt
'''

for iRep in arange(n_reps):

    # set random parameters
    # input_factor = normal(5, 1.25)
    # PYRs_sigma = normal(15, 1)
    # IN_sigma = normal(12.5, 1)
    # AMPA_mod = normal(0.7, 0.7/4)
    # GABA_mod = normal(2, 0.5)
    # connectivity = normal(0.25, 0.25/4)
    # lognormal_syn_weight = rand(1)
    # lognormal_syn_number = rand(1)
    # prop_syn_number = rand(1)

    input_factor = 5
    PYRs_sigma = 15
    IN_sigma = 12.5
    AMPA_mod = 0.85
    GABA_mod = 2
    connectivity = 0.25
    lognormal_syn_weight = 0.6
    lognormal_syn_number = 0.6
    prop_syn_number = 0.6

    PYRsProp = uniform(0.05, 0.95)
    nPYRS = int(n_neurons * PYRsProp)
    nINs = int(n_neurons - nPYRS)

    ########### define neuron groups ###########
    PYRs = NeuronGroup(nPYRS, method='euler',
                       model=eqsPYR,
                       threshold="V>Vthr", reset="V=Vrest",
                       refractory=refractoryE)
    PYRs.Cm = CmE
    PYRs.tau = CmE / gLeakE
    PYRs.sigma = PYRs_sigma * mV

    IN = NeuronGroup(nINs, method='euler',
                     model=eqsIN,
                     threshold="V>Vthr", reset="V=Vrest",
                     refractory=refractoryI)
    IN.Cm = CmI
    IN.tau = CmI / gLeakI
    IN.sigma = IN_sigma * mV

    # define AMPA and GABA synapses parameters
    Cee = Synapses(PYRs, PYRs, 'w: 1', on_pre='gea+=w')
    Cei = Synapses(PYRs, IN, 'w: 1', on_pre='gea+=w')
    Cie = Synapses(IN, PYRs, 'w: 1', on_pre='gi+=w')
    Cii = Synapses(IN, IN, 'w: 1', on_pre='gi+=w')

    # compute all connectivity matrices
    EE_mat = get_bin_mat(nPYRS, nPYRS, connectivity, NaN, NaN, lognormal_syn_number, prop_syn_number)
    EI_mat = get_bin_mat(nPYRS, nINs, connectivity, NaN, NaN, lognormal_syn_number, prop_syn_number)
    IE_mat = get_bin_mat(nINs, nPYRS, connectivity, NaN, NaN, lognormal_syn_number, prop_syn_number)
    II_mat = get_bin_mat(nINs, nINs, connectivity, NaN, NaN, lognormal_syn_number, prop_syn_number)

    # check that there were no issues
    tot_nans = sum(sum(isnan(EE_mat)) + sum(isnan(EI_mat)) + sum(isnan(IE_mat)) + sum(isnan(II_mat)))

    # only proceed if all the matrices are okay
    if tot_nans == 0:

        # now connect all synapses
        # PYR to PYR    
        sources, targets = EE_mat.nonzero()
        Cee.connect(i=sources, j=targets)
        Cee.delay = 0 * ms
        # PYR to IN    
        sources, targets = EI_mat.nonzero()
        Cei.connect(i=sources, j=targets)
        Cei.delay = 0 * ms
        # IN to PYR    
        sources, targets = IE_mat.nonzero()
        Cie.connect(i=sources, j=targets)
        Cie.delay = 0 * ms
        # IN to IN    
        sources, targets = II_mat.nonzero()
        Cii.connect(i=sources, j=targets)
        Cii.delay = 0 * ms

        # compute synaptic weight distributions
        if lognormal_syn_weight > 0.5:
            gEE = lognormal(0, 1, Cee.w.shape[0]) / 25 * AMPA_mod
            gEI = lognormal(0, 1, Cei.w.shape[0]) / 25 * AMPA_mod
            gIE = lognormal(0, 1, Cie.w.shape[0]) / 6 * GABA_mod
            gII = lognormal(0, 1, Cii.w.shape[0]) / 30 * GABA_mod
        else:
            gEE = normal(sqrt(e), 0.5, Cee.w.shape[0]) / 25 * AMPA_mod
            gEI = normal(sqrt(e), 0.5, Cei.w.shape[0]) / 25 * AMPA_mod
            gIE = normal(sqrt(e), 0.5, Cie.w.shape[0]) / 6 * GABA_mod
            gII = normal(sqrt(e), 0.5, Cii.w.shape[0]) / 30 * GABA_mod

        # set the weight for all the synapses
        Cee.w = gEE
        Cei.w = gEI
        Cie.w = gIE
        Cii.w = gII

        # initialize voltage
        PYRs.V = Vrest + (rand(nPYRS) * 5 - 5) * mV
        IN.V = Vrest + (rand(nINs) * 5 - 5) * mV

        # record spikes of excitatory neurons
        Sp_E = SpikeMonitor(PYRs, record=True)
        # record spikes of inhibitory neurons
        Sp_I = SpikeMonitor(IN, record=True)
        # record voltage
        Vm_E = StateMonitor(PYRs, 'V', record=True, clock=voltage_clock)
        Vm_I = StateMonitor(IN, 'V', record=True, clock=voltage_clock)
        # record exc. & inh. currents at E
        gE = StateMonitor(PYRs, 'gea', record=True, clock=voltage_clock)
        gI = StateMonitor(PYRs, 'gi', record=True, clock=voltage_clock)

        # ------------------------------------------------------------------------------
        # Run the simulation
        # ------------------------------------------------------------------------------
        run(simulation_time)

        dict2save = {}
        dict2save['spikes_PYR'] = Sp_E.i, Sp_E.t
        dict2save['spikes_IN'] = Sp_I.i, Sp_I.t
        dict2save['input_factor'] = input_factor
        dict2save['PYRs_sigma'] = PYRs_sigma
        dict2save['IN_sigma'] = IN_sigma
        dict2save['AMPA_mod'] = AMPA_mod
        dict2save['GABA_mod'] = GABA_mod
        dict2save['connectivity'] = connectivity
        dict2save['lognormal_syn_weight'] = lognormal_syn_weight
        dict2save['lognormal_syn_number'] = lognormal_syn_number
        dict2save['prop_syn_number'] = prop_syn_number

        savemat(root_dir + 'mat//' + str(v) + '_iRep_' + str(iRep) + '.mat', dict2save)

        # IP: save things in py format - spike trains only
        file_name_py = root_dir + 'py//'

        np.save(file_name_py + str(v) + '_iRep_' + str(iRep) + '_spikes_e', Sp_E.i)
        np.save(file_name_py + str(v) + '_iRep_' + str(iRep) + '_spikes_times_e', Sp_E.t / ms)

        np.save(file_name_py + str(v) + '_iRep_' + str(iRep) + '_spikes_i', Sp_I.i)
        np.save(file_name_py + str(v) + '_iRep_' + str(iRep) + '_spikes_times_i', Sp_I.t / ms)

        spike_trains_e = Sp_E.spike_trains()  # dict
        spike_trains_i = Sp_I.spike_trains()  # dict

        with open(file_name_py + str(v) + '_iRep_' + str(iRep) + '_spike_train_e', 'wb') as handle_e:
            pickle.dump(spike_trains_e, handle_e)

        with open(file_name_py + str(v) + '_iRep_' + str(iRep) + '_spike_train_i', 'wb') as handle_i:
            pickle.dump(spike_trains_i, handle_i)

        print('done with simulation ' + str(iRep))

        del Sp_E, Sp_I
    else:
        disp('problems with some matrices, skipping this simulation')
