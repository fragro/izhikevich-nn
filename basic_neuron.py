#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Frank Grove on 2009-10-21.
Copyright (c) 2009 Fragro Inc. All rights reserved.
"""

import sys,os,math,time
import random as rand
from numpy import *
import re
#import matplotlib.pyplot as plt


class simulation(object):
    """Simulates the Izhikevich model showing the effects of bursting neurons"""
    def __init__(self, a1, b1, c1, d1, current, time):
        #Simulation parameters
        self.timescale = time #number of milliseconds to simulate
        self.n = 1 #number of neurons running in the simulation
        self.a = array([a1 for i in range(self.n)])     #timescale of the recovery variable
        self.b = array([b1 for i in range(self.n)])     #sensitivity of the recovery variable
        self.c = array([c1 for i in range(self.n)])     #mV -- after-spike reset of the membrane potential
        self.d = array([d1 for i in range(self.n)])     #after-spike reset of the recovery variable u
        init_v = c1
        init_u = self.b[0] * init_v
        self.nList = array([basic_neuron(init_v, init_u) for i in range(self.n)])
        self.inp = array([0 for i in range(self.n)])
        self.current = current
        self.fired = []
        
    def neural_sim(self, trainfile, testfile):
        #initialize environment; inputs and outputs from the data
        neurons = 4100
        time = 500
        excite = int(.8 * neurons)
        inhib = int(.2 * neurons)
        num_neurons_per_feature = 50
        num_neurons_per_input = 1
        tao_c = 500.0 #time constant for the STDP
        tao_d = 100.0  #time constant for the dopamine
        reward_wait_time = 60
        stimulus_interval = 100
        class0 = 0
        class1 = 0
        
        neural_input, inputset, eligible_neurons, classes = initInput(excite, trainfile, num_neurons_per_input)
        neural_output = initOutput(eligible_neurons, classes, num_neurons_per_feature)
        
        probs_file = open('probs.txt', 'w')
        dopa_file = open('dopa.txt', 'w')
        c_file = open('c_file.txt', 'w')
        
        
        #print neural_input
        #print inputset
        #print 'classes',classes
        #print neural_output
        
        #initialize the neural network
        timescale = time #number of milliseconds to simulate
        rand_exc = random.random(excite)
        rand_inh = random.random(inhib)
        a = [0.02*ones(excite), 0.02+0.08*rand_inh]         #timescale of the recovery variable
        b = [0.2*ones(excite), 0.25-0.05*rand_inh]          #sensitivity of the recovery variable
        c = [-65+15*pow(rand_exc,2), -65*ones(inhib)]       #mV -- after-spike reset of the membrane potential
        d = [8-6*pow(rand_exc,2), 2*ones(inhib)]
        testEXC = [[pnt(0.5) for i in range(excite)] for j in range(excite+inhib)] #each node has prob*800 synapses that are excitatorty
        testINH = [[pnt(-1.0) for i in range(inhib)] for j in range(excite+inhib)] #each node has prob*200 synapses that are inhibitorty
        synapse_index = []
        for i in range(len(testEXC)): #need to grab the active percentage of synapses to reduce computational load on STDP
            active_syn = []
            for j in range(len(testEXC[i])):
                if testEXC[i][j] > 0.0: active_syn.append(j)
            synapse_index.append(active_syn)
        S = [array(testEXC), array(testINH)] #synapse array
        cnt = 0
        v = -65*ones(excite+inhib)
        u = v.copy()
        u *= b[0][0]
        
        #we only need to modify the strength of the excitatory synapses: the inhibitory remain plastic
        c_ = [zeros(excite) for i in range(excite+inhib)] #actiavtion of some relatively slow synaptic tag at neuron i
        dopa = .001 #level of dopamine at neuron i
        
        #Need some times for stimulus to occur
        stimulus_times = range(100,timescale,stimulus_interval)
        
        firetimes = zeros(excite+inhib)
        fired = []
        ind_fired = []
        firings = []
        reward_time = -1
        dopamine_time = -1
        input_delivered = False
        dopa_times = []
        last_stim_time = -1
        class0_record = []
        class1_record = []
        eligibility_trace = -1 #used to identify a node
        c_pre = -1
        c_post = -1
        E_I = empty(excite)
        I_I = empty(inhib)
        #I_I2 = empty(inhib)
        #E_I2 = empty(excite)
        last_input = None
        for t in range(timescale): # for each timestep in the timescale of the universe
            #print t, ': Dopa : ', dopa
            ###This will soon get input from a instructive source
            inp, j = getInput(t, stimulus_times, neural_input, inputset, excite)
            if inp != None and not input_delivered:
                dopamine_time = t+ rand.randint(20,reward_wait_time)
                last_stim_time = t
                last_input = inp
                dopa_times.append(dopamine_time)
                class0_record.append(class0)
                class1_record.append(class1)
                class0 = 0.0
                class1 = 0.0
            standard = [ 5 * random.random(excite) , 2.5 * random.random(inhib)]
            if (j == None and not input_delivered) or input_delivered:
                I = standard
            elif j != None and not input_delivered: 
                input_delivered = True
                I = [ j + 5, 2.5 * random.random(inhib)]                
                #print '*'*20 , 'Input Stimulus: ', t
            if t == 0: fired = array(map(lambda x: x>=30 , v[0:excite+inhib])) #determine which neurons are firing at this time step
            for i in range(excite+inhib): 
                if fired[i]: firetimes[i] = t #update fire times
            firings.append(fired) #create paper trail for firings
            for i in range(excite): #for each excitatory neuron we need to determine if it fired, update plasticity, etc.
                if fired[i]: #reset the neuron and send message that it fired
                    v[i] = c[0][i]
                    u[i] = u[i] + d[0][i]
                #E_I2[i] = sum(S[1][i] * fired[excite:excite+inhib]) #
                E_I[i] = sum(S[0][i] * fired[0:excite]) + sum(S[1][i] * fired[excite:excite+inhib]) #Calculate the total input for this Excitatory neuron
            #print E_I
            I[0] = E_I + I[0]
            for i in range(inhib): 
                if fired[i+excite]: 
                    v[i+excite] = c[1][i]
                    u[i+excite] = u[i+excite] + d[1][i]
                I_I[i] = sum(S[0][i+excite] * fired[0:excite]) + sum(S[1][i+excite] * fired[excite:excite+inhib]) #Calculate the total input for the inhibitory neuron
            #print t, ': ' , sum(E_I2), ' ' , sum(I_I2)
            #print sum(I_I)
            I[1] = I_I + I[1]
            v[0:excite] += 0.5 * ( 0.04 * pow(v[0:excite],2) + 5 * v[0:excite] + 140 - u[0:excite] + I[0]) #incremenet the voltage potential of the excitatory neurons .5
            v[0:excite] += 0.5 * ( 0.04 * pow(v[0:excite],2) + 5 * v[0:excite] + 140 - u[0:excite] + I[0]) #incremenet the voltage potential of the excitatory neurons .5
            v[excite:excite+inhib+1] += 0.5 * ( 0.04 * pow(v[excite:excite+inhib],2) + 5 * v[excite:excite+inhib] + 140 - u[excite:excite+inhib] + I[1])# inc. inhib. neurons
            v[excite:excite+inhib] += 0.5 * ( 0.04 * pow(v[excite:excite+inhib],2) + 5 * v[excite:excite+inhib] + 140 - u[excite:excite+inhib] + I[1])# inc. inhib. neurons
            u[0:excite] = u[0:excite] + a[0] * (b[0] * v[0:excite] - u[0:excite])
            u[excite:excite+inhib] = u[excite:excite+inhib] + a[1] * (b[1] * v[excite:excite+inhib] - u[excite:excite+inhib])
            #record infividual neuron
            ind_fired.append(map(lambda x: min(30, x), v)) #record all neuron voltage potentials to record spiking behavior
            if t != 0: fired = array(map(lambda x: x>=30 , v[0:excite+inhib]))
            #print I
            #print 'update synapses'
            ### dc = –c/tao + STDP(t) * delta(t - tpre/tpost)
            #Now we update the synapses with dopamine/STDP induced weight change
            total_synaptic_strength = 0.0
            total_synapses = 0.0
            total_stdp = 0.0
            cnt_stdp = 0.0
            total_c = 0.0
            num_c = 0.0
            max_c = 0.0
            max_c_syn = 0.0
            for i in range(excite): #for each presynaptic neuron
                for syn_ind in synapse_index[i]: #corresponds to post synaptic neuron with synapse strength syn
                    if firetimes[syn_ind] != 0.0 and firetimes[i] != 0.0:
                        syn = S[0][i][syn_ind]
                        total_synaptic_strength+=syn
                        total_synapses+=1
                        tao = firetimes[syn_ind] - firetimes[i]
                        #print 'stdp ', STDP(tao,S[0][i][syn_ind]) 
                        #print 'delta ',  delta(t - (firetimes[syn_ind]+firetimes[i])/2) 
                        #total_stdp += STDP(tao,S[0][i][syn_ind])
                        #print float(-c_[i][syn_ind])/tao_c
                        dc = -c_[i][syn_ind]/tao_c + STDP(tao,S[0][i][syn_ind]) * delta(t,firetimes[syn_ind],firetimes[i]) #taking the average, notation not clear
                        c_[i][syn_ind] = c_[i][syn_ind] + dc #update c
                        total_c+=c_[i][syn_ind]
                        if eligibility_trace == -1 and c_[i][syn_ind] > max_c:
                            max_c =c_[i][syn_ind]
                            c_pre = i
                            c_post = syn_ind
                        elif eligibility_trace == 1 and i == c_pre and syn_ind == c_post:
                            max_c =c_[i][syn_ind]
                            max_c_syn = S[0][i][syn_ind]
                        num_c+=1
                        ds = c_[i][syn_ind] * dopa
                        S[0][i][syn_ind] += ds
                        S[0][i][syn_ind] = min(max(S[0][i][syn_ind],0),4)
            #record the c of a single synaptic link
            if max_c > .1 and c_pre != -1 and eligibility_trace == -1:
                #print 'pre: ', c_pre , ' ;;;post: ' , c_post
                #print max_c
                #print max_c_syn
                eligibility_trace = 1
            elif eligibility_trace == 1:
                c_file.write(str(t) + '\t' + str(c_pre) +'\t' + str(c_post) + '\t' + str(max_c) + '\t' + str(max_c_syn)  + '\n')
            if t % 20 == 0 and num_c != 0: 
                #print 'mean c: ', total_c/num_c ,  ' dopa: ', dopa
                """ numFired=0
                for i in fired:
                    if int(i) == 1:
                        numFired+=1
                print 'Fired: ' , numFired"""
                #if total_synapses!=0.0: print 'mean synapse: ', total_synaptic_strength/total_synapses
            #print 't: ', t,'  ...dopa: ' , dopa
            
            
            ##Determine motor neuron output
            if last_input != None:
                inc0,inc1 = checkOutput(fired[0:excite], neural_output, last_input[0])
                if t < last_stim_time + 20: #20 ms window for checking groups 1 vs 0
                    class0+= inc0
                    class1+=inc1
                elif t == last_stim_time + 21:
                    tot = float(class0 + class1)
                    if tot != 0:
                        cl1p = class0 / tot
                        cl2p = class1 / tot
                    else:
                        cl1p = .5
                        cl2p = .5
                    probs_file.write(str(t) + ' \t' + str(cl1p)+ ' \t' + str(cl2p) + '\n')
                reward = False
                if dopa_times.count(t) > 0:
                    ind = dopa_times.index(t)
                    cl1 = class1_record.pop(ind)
                    cl0 = class0_record.pop(ind)
                    dopa_times.remove(t)
                    corr_class = last_input[0]
                    if int(corr_class) == 0 and cl0 > cl1:
                        reward = True
                    if int(corr_class) == 1 and cl1 > cl0:
                        reward = True
                    print '*' * 30
                    print 'Reward Time: ', reward, ' that dopa Injected : class== ', last_input[0]
                    print 'Class0: ', cl0
                    print 'Class1: ', cl1
                    input_delivered = False
                if reward:
                    reward_time = t
                ##Update Dopamine levels based on success with output
                if reward_time == t:
                    reward = True
                else:
                    reward = False
                dopa += -dopa / tao_d + DA(t, reward_time, reward)
                dopa = max(dopa, 0.0)
                dopa_file.write(str(t) + ' \t' + str(dopa) + '\n')

        #final synapse dist.
        syn_num = 0
        syn_weight = []
        s = open('synapse_dist.txt', 'w')
        for i in range(excite):
            #for each presynaptic neuron
            for syn_ind in synapse_index[i]:
                #corresponds to post synaptic neuron with synapse strength syn
                syn_weight.append(S[0][i][syn_ind])
                s.write(str(syn_num) + '\t' + str(S[0][i][syn_ind]) + '\n')
                syn_num += 1
        s.close()
        #graphHist(syn_weight,100,'Synapse Dist.','mV','blue')
        dopa_file.close()
        c_file.close()
        probs_file.close()
        firings_file = open('firing_file.txt', 'w')
        for i in xrange(0, len(firings)):
            rec = firings[i]
            for j in range(len(rec)):
                if rec[j] == True:
                    firings_file.write(str(i) + '\t' + str(j) + '\n')
        firings_file.close()
        f = open('fire_info_net.txt', 'w')
        for timestep in range(len(ind_fired)):
            f.write(str(timestep) + '\t' + "\t".join([str(i) for i in ind_fired[timestep]]) + '\n')
        f.close()


def graphHist(l, numbins, xlabel, ylabel, color):
    n, bins, patches = plt.hist(l, numbins,  facecolor=color)   # add a 'best fit' line
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()


class basic_neuron(object):
    """Class for the basic Izhikevich neuron model. This model combines the biologically realistic nature of Hodgkins-Huxley with
            computationally efficiene ODEs to allow for simulation of large amounts of neurons"""
    def __init__(self, v, u):
        self.v = v
        #membrane potential of neuron
        self.u = u
        #membrane recovery variable

    def updateV(self, I):
        #step .5 ms for numerical stability
        self.v += 0.5 * (0.04 * math.pow(self.v, 2) + 5 * self.v + 140 - self.u + I)
        self.v += 0.5 * (0.04 * math.pow(self.v, 2) + 5 * self.v + 140 - self.u + I)

    def updateU(self, a, b):
        global globSim
        if globSim == 2:
            self.u += a * b * (b * self.v + 65)
        else:
            self.u += a * (b * self.v - self.u)

    def resetNode(self, c, d):
        if self.v >= 30:
            self.v = c
            self.u = self.u + d


#returns the indexes of the features as mapped onto the neural network and inputset of data, and remaing neurons
def initInput(excite, trainfile, num_neurons_per_feature):
    inputset = []
    f = open(trainfile, 'r')
    features_set = False
    neural_input = []
    classes = []
    for i in f:
        j = split(i.strip('\n').strip('\r'))
        eligible_neurons = range(excite)
        if not features_set:
            for binary in j[1:]:  # now we get the indeex of x number random neurons for each of the features
                feature = []
                for i in range(num_neurons_per_feature):
                    feature.append(eligible_neurons.pop(rand.randint(0, len(eligible_neurons) - 1)))
                neural_input.append(feature)
        if classes.count(j[0]) == 0:
            classes.append(j[0])
        features_set = True
        inputset.append([j[0], j[1:]])
    #return neural_input, inputset, eligible_neurons, classes
    return neural_input, inputset, eligible_neurons, [1, 0]


def initOutput(eligible_neurons, classes, num_neurons_per_feature):
    neural_output = []
    for i in classes:
        feature = []
        for j in range(num_neurons_per_feature):
            feature.append(eligible_neurons.pop(rand.randint(0, len(eligible_neurons) - 1)))
        neural_output.append((i, feature))
    return neural_output


def getInput(t, stimulus_times, neural_input, inputset, excite):
    if stimulus_times.count(t) == 1:
        stimulus_times.remove(t)
        r = rand.randint(0, len(inputset) - 1)
        inp = inputset[r]
        arr = zeros(excite)
        for i in range(len(inp[1])):
            #for each feature in this input set
            if int(inp[1][i]) == 1:
                for k in neural_input[i]:
                    arr[k] = arr[k]+ 10
        return inp, arr
    else: return None, None


#check neuron output against desired output
def checkOutput(fired, neural_output, clas):
    check = True
    inc0 = 0
    inc1 = 0
    for j in neural_output:
        for i in j[1]:
            if fired[i] == True:
                if int(j[0]) == 0:
                    inc0 += 1
                if int(j[0]) == 1:
                    inc1 += 1
    return inc0, inc1


def split(line):
    pattern = re.compile(r'\s*("[^"]*"|.*?)\s*,')
    return [x[1:-1] if x[:1] == x[-1:] == '"' else x
        for x in pattern.findall(line.rstrip(',') + ',')]


def pnt(x):
    if rand.random() < .2:
        return x * rand.random()
    else:
        return 0.0


def DA(t, reward_time, reward):
    if reward:
        return 0.5 / 10.0
    else:
        return 0.0
        # tonic concentration of dopamine


#STDP Hebbian correlation based synaptic plasticity update
def STDP(tao, syn):
    n_plus = .3
    n_minus = 3.0
    if tao > 0:
        A = (4 - syn) * n_plus
        return A * exp(-tao / 10.0)
    elif tao < 0:
        A = (syn) * n_minus
        return -A * exp(tao / 10.0)
    else:
        return 0.0

#Dirac delta function: may need different a value
def delta(time1, firei, firej):
    if time1 % 10 == 0:
        if time1 - firei < 10 and time1 - firej < 10:
            return 0.1
        else:
            return 0.0
    else:
        return 0.0

def main():
    global globSim
    current = 10
    timescale = 1000
    neuron_map = {0: 'Intrinsically Bursting', 1: 'Chattering', 2: 'Spike Accomodation', 3: 'Regular Spiking', 4: 'Fast Spiking'}
    ####################################
    #### 0 = intrinsically bursting ####
    #### 1 = chattering             ####
    #### 2 = spike accomodation     ####
    #### 3 = regular spiking        ####
    #### 4 = fast spiking           ####
    ####################################
    params = [[.02,.2,-55.0,4.0], [.02,.2,-50.0,2.0], [.02,1,-55.0,4.0], [.02, .2, -65, 8.0], [.1, .2, -65, 2.0]]
    for i in range(len(params)):
        print 'Running Simulation of a single \t' , neuron_map[i], 'for ', timescale , ' ms'
        globSim = i
        oldtime = time.clock()
        sim = simulation(params[globSim][0], params[globSim][1], params[globSim][2], params[globSim][3], current, timescale)
        #sim.go()
        newtime = time.clock()
        timedelta = newtime - oldtime
        print 'Real Time: ', timedelta
    print 'Running Sparsely Connected Network of Excitatory/Inhibitory Neurons'
    datafile = 'basic.train'
    testfile = 'SPECT.test.txt'
    sim.neural_sim(datafile, testfile)


if __name__ == '__main__':
    main()

