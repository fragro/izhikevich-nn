#!/usr/bin/python
# Import all the necessaries
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.driver import device_attribute as A
from DemoMetaMatrixmulCheetah import matrixmul_opt
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel
from pycuda.curandom import rand as curand

import pprint
import argparse

import numpy
import numpy.linalg as la

import sys
import re
import random as rand
pp = pprint.PrettyPrinter(indent=4)

def split(line):
    pattern = re.compile(r'\s*("[^"]*"|.*?)\s*,')
    return [x[1:-1] if x[:1] == x[-1:] == '"' else x
        for x in pattern.findall(line.rstrip(',') + ',')]



###THIS FUNCTION ALSO NEEDS TO BE REDONE AS A KERNEL
def getInput(t, stimulus_times, neural_input, inputset, excite):
    if stimulus_times.count(t) == 1:
        stimulus_times.remove(t)
        r = rand.randint(0, len(inputset) - 1)
        inp = inputset[r]
        arr = numpy.zeros(excite)
        for i in range(len(inp[1])):
            #for each feature in this input set
            if int(inp[1][i]) == 1:
                for k in neural_input[i]:
                    arr[k] = arr[k]+ 10
        return inp, arr
    else: return None, None

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


#### PARSE ARGS #### 
parser = argparse.ArgumentParser(description="Simulate a network of neurons "+
    "with small-world connectivity.")
parser.add_argument('num_neurons', metavar='neurons', type=int,
    help='the number of neurons to simulate')
parser.add_argument('sim_length', metavar='length', type=int,
    help='the number of time points to simulate')
parser.add_argument('-r', '--random', action='store_true',
    help='generate a randomly connected network')

args = parser.parse_args()
num_neurons  = args.num_neurons
sim_length = args.sim_length
#### /PARSE ARGS #### 

"""
### Thread/block management will have to be figured out with benchmarking
### (like the pros do)
# Figure out how to manage threads/blocks
dev = cuda.Device(0)
attr = dev.get_attributes()
num_MPs = attr[A.MULTIPROCESSOR_COUNT]
max_threads = attr[A.MAX_THREADS_PER_MULTIPROCESSOR]*num_MPs
num_neurons
"""

#### INITIALIZATION ####
# Set up necessary arrays
# Internal neuron variables. Synaptic weights and c.
# in weights, rows are presyn & cols are psotsyn
try:
    weights = numpy.ones((num_neurons, num_neurons)).astype(numpy.float32)
except ValueError:
    sys.stderr.write("There is not enough memory for %i neurons. Try again with fewer neurons.\n"%(num_neurons))
c = numpy.ones((1, num_neurons)).astype(numpy.uint32)

# fire events - these are boolean but use floats to avoid type conflicts
fired = numpy.zeros((1, num_neurons)).astype(numpy.float32)
# fire time
fire_time = numpy.ones((1, num_neurons)).astype(numpy.uint32)
#threadid - provided by CUDA

#dopamine
dopamine = 0
excite = int(.8 * num_neurons)
inhib = int(.2 * num_neurons)
num_neurons_per_feature = 50
num_neurons_per_input = 1
tao_c = 500.0 #time constant for the STDP
tao_d = 100.0  #time constant for the dopamine
reward_wait_time = 500
stimulus_interval = 500
class0 = 0
class1 = 0
trainfile = 'basic.train'
input_delivered = False

"""
#initialize the neural network
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
u = v.copy()
u *= b[0][0]
"""
rand_exc = curand(excite, dtype=numpy.float32)
rand_inh = curand(inhib, dtype=numpy.float32)

v = -65 * numpy.ones(excite+inhib)
v = gpuarray.to_gpu(v)

a_exc = 0.02*numpy.ones(excite)
a_exc = gpuarray.to_gpu(a_exc)

a_inh = 0.02+0.08 * rand_inh
#timescale of the recovery variable

b_exc = 0.2 * numpy.ones(excite)
b_exc = gpuarray.to_gpu(b_exc)
b_inh = 0.25-0.05 * rand_inh

#sensitivity of the recovery variable
c_exc = -65+15 * pow(rand_exc,2)
c_inh = -65 * numpy.ones(inhib)
c_inh = gpuarray.to_gpu(c_inh)


#mV -- after-spike reset of the membrane potential
d_exc = 8-6*pow(rand_exc,2)
d_inh = 2 * numpy.ones(inhib)
d_inh = gpuarray.to_gpu(d_inh)


#### /INITIALIZATION ####
neural_input, inputset, eligible_neurons, classes = initInput(excite, trainfile, num_neurons_per_input)
#Need some times for stimulus to occur
stimulus_times = range(100,sim_length,stimulus_interval)
pp.pprint(stimulus_times)


###KERNEL FOR DETERMINGING THE INPUT
random_input_comb = ElementwiseKernel(
        "float a, float *x",
        "x[i] = a*x[i]",
        "linear_combination")

set_input_comb = ElementwiseKernel(
        "float j, float h, float *x",
        "x[i] = h+j",
        "linear_combination")

#### MAIN LOOP ####
for i in xrange(sim_length):
    # Calculate input
    # Input for a neuron is the sum of the synaptic weights of all neurons that
    # fired onto it
    #### also need to include input from basic.train
    inp, arr = getInput(i, stimulus_times, neural_input, inputset, excite)
    if inp != None:
      print arr


    inputs = matrixmul_opt(fired, weights)[0][0]

    ################################################################################
    if inp != None and not input_delivered:
        dopamine_time = i + rand.randint(20,reward_wait_time)
        last_stim_time = i
        last_input = inp
        dopa_times.append(dopamine_time)
        class0_record.append(class0)
        class1_record.append(class1)
        class0 = 0.0
        class1 = 0.0

    # should break this down into a kernel as well
    # determining the input is a bit more complex 
    inhibitory = curand((inhib), dtype=numpy.float32)
    random_input_comb(2.5, inhibitory)

    if (arr == None and not input_delivered) or input_delivered:
        excitatory = curand((excite), dtype=numpy.float32)
        random_input_comb(5, excitatory)

    elif arr != None and not input_delivered: 
        input_delivered = True
        set_input_comb(arr, 5, excitatory)

    #### Input is ready to go, now we need to calculate the total inputs/Izekivich model

    if i == 0: fired = numpy.array(map(lambda x: x>=30 , v[0:excite+inhib])) #this is lazy but it's only one time steo

    #update firetimes



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
#### /MAIN LOOP ####    

#for neu,inp in enumerate(inputs):
#    print "Neuron %i has an input of %i."%(neu,inp)

"""
kernels list:

	input kernel
	fire kernel
	fire times: check if a neuron is above the threshold, register fire 
        in array
	stdp: adjust internal C value based on STDP
	array of length N for neuron classes, registering output


***   arrays list:

   2D synaptic strength
   fire times
   fire boolean
   input (combinations of fire_boolean*syanptic strength) + stimuli
   a = timescale of the recovery variable
   b = sensitivity of the recovery variable
   c = after-spike reset of the membrane potential
   d = after-spike reset of the recovery variable u
   v = nueron voltage
   u = recovery variable
"""
