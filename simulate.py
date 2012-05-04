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
from create_smallworld import create_smallworld

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


def DA(t, reward_time, reward):
    if reward:
        return 0.5 / 10.0
    else:
        return 0.0

###THIS FUNCTION ALSO NEEDS TO BE REDONE AS A KERNEL
def checkOutput(fired, neural_output):
    val = fired * neural_output[0]
    inc0 = sum(val.get().flatten())
    val = fired * neural_output[1]
    inc1 = sum(val.get().flatten())
    return inc0, inc1


###THIS FUNCTION ALSO NEEDS TO BE REDONE AS A KERNEL
def getInput(t, stims, classes):
    if stims[t] != -1:
        clss = int(stims[t])
        return classes[clss] * 10, clss
    else:
        return None, None


def initOutput(eligible_neurons, classes, num_neurons_per_feature):
    neural_output = {}
    for i, v in classes.items():
        binarr = numpy.zeros(excite+inhib)
        for j in range(num_neurons_per_feature):
            idx = eligible_neurons.pop(rand.randint(0, len(eligible_neurons) - 1))
            binarr[idx] = 1
        neural_output[i] = gpuarray.to_gpu(binarr)
    return neural_output

#returns the indexes of the features as mapped onto the neural network and inputset of data, and remaing neurons
def initInput(excite, trainfile, num_neurons_per_input):
    f = open(trainfile, 'r')
    classes = {}
    for i in f:
        j = split(i.strip('\n').strip('\r'))
        eligible_neurons = range(excite)
        for binary in j:
            if int(binary) not in classes:
                binarr = numpy.array([[0] for i in xrange(excite)])
                for i in range(num_neurons_per_input):
                    idx = eligible_neurons.pop(rand.randint(0, len(eligible_neurons) - 1))
                    binarr[idx] = [1]
                classes[int(binary)] = binarr 
    return classes, eligible_neurons


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

#dopamine
synapses_per = num_neurons / 10
dopamine = 0
excite = int(.8 * num_neurons)
inhib = int(.2 * num_neurons)
num_neurons_per_feature = num_neurons / 50
num_neurons_per_input = num_neurons / 50
tau_c = 500.0 #time constant for the STDP
tau_d = 100.0  #time constant for the dopamine
reward_wait_time = 60
reward_time = -1        
stimulus_interval = 100
class0 = 0
class1 = 0
trainfile = 'basic.train'
dopa_times = []
class0_record = []
class1_record = []
last_input = None
# Set up necessary arrays
# Internal neuron variables. Synaptic weights and c.
# in weights, rows are presyn & cols are psotsyn
try:
    #weights_cpu = numpy.array([[rand.random() for k in xrange(num_neurons)]  for h in xrange(num_neurons)]).astype(numpy.float32)
    weights_cpu = numpy.array(
        create_smallworld(num_neurons, synapses_per, print_stats=True)).astype(numpy.float32)
    weights = gpuarray.to_gpu(weights_cpu)
except ValueError:
    sys.stderr.write("There is not enough memory for %i neurons. Try again with fewer neurons.\n"%(num_neurons))

# fire events - these are boolean but use floats to avoid type conflicts
fired_cpu = numpy.array([[0] for i in xrange(num_neurons)]).astype(numpy.float32)

#TESTING
fired_cpu = numpy.array([[rand.randint(0,1)] for i in xrange(num_neurons)]).astype(numpy.float32)
fired = gpuarray.to_gpu(fired_cpu)

inputs = gpuarray.zeros((num_neurons,num_neurons), numpy.float32)


input_delivered = False

rand_exc = curand(excite, dtype=numpy.float32)
rand_inh = curand(inhib, dtype=numpy.float32)
firetimes = gpuarray.zeros_like(fired)

v = numpy.array([[-65] for i in xrange(excite+inhib)])
u = v.copy()
v = gpuarray.to_gpu(numpy.array(v).astype(numpy.float32))

#for testing purposes, never changes
testbay = gpuarray.to_gpu(numpy.array([[30] for i in xrange(num_neurons)]))

a = [[0.02] for i in xrange(excite)]
a.extend([[0.02+0.08*rand.random()] for i in xrange(inhib)])
a = gpuarray.to_gpu(numpy.array(a).astype(numpy.float32))

#timescale of the recovery variable
b = [[0.2] for i in xrange(excite)]
b.extend([[0.25-0.05*rand.random()] for i in xrange(inhib)])
u = gpuarray.to_gpu(u * 0.02)
b = gpuarray.to_gpu(numpy.array(b).astype(numpy.float32))

c = [[-65+15*pow(rand.random(),2)] for i in xrange(excite)]
c.extend([[-65] for i in xrange(inhib)])
c = gpuarray.to_gpu(numpy.array(c).astype(numpy.float32))
#sensitivity of the recovery variable


#post = [[0 for i in xrange(num_neurons)] for j in xrange(synapses_per)]
pe_left = [synapses_per]*num_neurons

#mV -- after-spike reset of the membrane potential
d = [[8-6*pow(rand.random(),2)] for i in xrange(excite)]
d.extend([[2] for i in xrange(inhib)])
d = gpuarray.to_gpu(numpy.array(d).astype(numpy.float32))


#### /INITIALIZATION ####
classes, eligible_neurons = initInput(excite, trainfile, num_neurons_per_input)

neural_output = initOutput(eligible_neurons, classes, num_neurons_per_feature)

#Need some times for stimulus to occur
stimulus_times = range(100, sim_length ,stimulus_interval)
stims = -numpy.ones(sim_length)
for i in stimulus_times:
    stims[i] = rand.randint(0,1)

pp.pprint(stimulus_times)

####CUDA SPECIFIC
BLOCK_X=4
BLOCK_Y=4
block_size = 16
work_size = 1

# DEBUG: Randomly populate post
# Generate list of postsynaptic connections
"""
for num,col in enumerate(post):
  valid = 0
  # For each prospective neuron
  it = 0
  while not valid:
    if it > 20:
      print "Could not generate synapses.",
      print" Adjust the number of neurons/synapses."
      sys.exit(1)

    # Get a list of neuron numbers
    q = range(num_neurons)
    # Remove this neuron from the list
    del q[num]
    # Randomize
    numpy.random.shuffle(q)

    # Prospective postsynaptic neuron list
    r = q[:synapses_per]

    # If the list in invalid, generate a new one
    for j in xrange(len(r)):
      if pe_left[r[j]] == 0:
        break
    valid = 1
    it += 1
  for j in xrange(len(r)):
    pe_left[r[j]] -= 1
  post[num] = r
"""
post = numpy.ones((num_neurons, synapses_per))
post *= -1
# I did this with list comprehensions first,
# but that doesn't produce a square array :(
for i in xrange(len(weights_cpu)):
    k = 0
    for j in xrange(len(weights_cpu[i])):
        if weights_cpu[i][j] != 0:
            post[i][k] = j
            k += 1

"""
# Check results
for i in xrange(len(post)):
    for j in post[i]:
        if weights_cpu[i][j] == 0 and j != -1:
            print "OH NOES"
            raw_input()
"""

print(len(post))
print(len(post[0]))
        
post = numpy.array(post).astype(numpy.uint32)
post_gpu = gpuarray.to_gpu(post)

# Time constant for dopamine
eligibility_trace = -1
#### /INITIALIZATION ####

#### STDP KERNEL ####
stdp_krnl = ElementwiseKernel(
    "float dopamine, float *c, float *fired, float *fire_time,"+
      "float *weights, float *post",
    """
    #define SYN_PER %(synapses_per)i
    #define NUM_NEU %(num_neurons)i
    #define TAU_C %(tau_c).3f

    float syn; 
    int both_fired;
    int tau;
    int cur_po;
    int old_c;
    int dc;
    
    /* FOR each postsynaptic neuron */
    for(int j = 0; j < SYN_PER; j++) {
      cur_po = post[j*NUM_NEU+i];
      syn = weights[cur_po*NUM_NEU+i];
      /* BEGIN IF */
      both_fired = (cur_po != -1 && fired[i]>0 && fired[cur_po]);
      tau = fire_time[cur_po]-fire_time[i];
      /* n_plus = 0.3, n_minus = 3.0 */
      old_c = c[cur_po*NUM_NEU+i];
      dc =  
      (
        fdividef(-old_c, TAU_C)
        + ((float) (tau > 0))
          *(4-syn)*0.3f
          *expf(fdividef(-tau, 10.0))
        + ((float) (tau < 0))
          *(-(syn)*3.0f)*expf(fdividef(-tau, 10.0))
      )
        *((float)both_fired);
      c[cur_po*NUM_NEU+i] = old_c + dc;
      syn += old_c + dc;
      weights[cur_po*NUM_NEU+i] = (syn > 4 || syn < 0)*syn + (syn > 4)*4;
    /* END FOR */
    }

    """%{"synapses_per":synapses_per, "tau_c":tau_c, 
          "num_neurons":num_neurons}
) 

#### MAIN LOOP ####
for t in xrange(sim_length):

    #print '-' * 100
    # Calculate input
    # Input for a neuron is the sum of the synaptic weights of all neurons that
    # fired onto it
    #### also need to include input from basic.train
    inp, clss = getInput(t, stims, classes)
    #print 'STIMETIMES: ' + str(inputset)

    ################################################################################
    if inp != None and not input_delivered:
        dopamine_time = t + rand.randint(20,reward_wait_time)
        last_stim_time = t
        last_input = inp
        last_class = clss
        dopa_times.append(dopamine_time)
        class0_record.append(class0)
        class1_record.append(class1)
        class0 = 0.0
        class1 = 0.0

    # should break this down into a kernel as well
    # determining the input is a bit more complex 
    inhibitory = numpy.array([[2.5 * rand.random()] for i in xrange(inhib)])

    if (inp == None and not input_delivered) or input_delivered:
        excitatory = [[5 * rand.random()] for i in xrange(excite)]

    elif inp != None and not input_delivered: 
        excitatory = inp
    
    excitatory = numpy.vstack((excitatory, inhibitory))


    #### Input is ready to go, now we need to calculate the total inputs/Izekivich model
    #    numpy.array(map(lambda x: x>=30 , v[0:excite+inhib])) 
    if t == 0:
        input_vector = gpuarray.to_gpu(excitatory)
        #MATRIX INPUT OP\
        fired = v > testbay

        #print fired * weights
    #update firetimes
    else:
        input_vector.set(excitatory)
        fired = v > testbay
        #print firetimes.shape
        #print fired.shape
        firetimes = -(fired - 1) * firetimes + t * fired


    v = fired * c + -(fired-1) * v
    u = fired * (u+d) + -(fired-1) * u

    #inputs = matrixmul_opt(weights, fired)
    #print inputs

    # get the kernel function from the compiled module
    # matrixmul = mod.get_function("MatrixMulKernel")

    # # call the kernel on the card
    # matrixmul(
    #     # inputs
    #     weights, fired, 
    #     # output
    #     inputs, 
    #     # grid of multiple blocks
    #     grid = (num_neurons / TILE_SIZE, num_neurons / TILE_SIZE),
    #     # block of multiple threads
    #     block = (TILE_SIZE, TILE_SIZE, 1), 
    # )

    new_inputs, gpu_time = matrixmul_opt(weights_cpu, fired_cpu)
    if t == 0:
      inputs = gpuarray.to_gpu(new_inputs.astype(numpy.float32))
    else:
      inputs.set(new_inputs.astype(numpy.float32))

    next_input = inputs + input_vector

    #print (fired * weights)
    #print weights_cpu[0][0]
    # print '-' * 80
    # print fired.shape
    # print weights.shape
    # #print (weights * fired) + input_vector
    # #inputs = (fired * weights) + input_vector
    # #print inputs == weights * fired #this is false always... random?
    # print inputs

    # # print  inputs_cpu - inputs.get()
    # #print "L2 norm:", la.norm(inputs_cpu - inputs.get())
    # #print weights_cpu * fired_cpu
    # #print sum(fired.get(numpy.empty((num_neurons), numpy.float32)))
    # #print pp.pprint(sum(inputs.get(numpy.empty((num_neurons,num_neurons), numpy.float32))[4]))
    # #print inputs.get(numpy.empty((num_neurons,num_neurons), numpy.float32))[0]
    # print v.shape
    # print u.shape
    # print inputs.shape
    #print v
    #print (b * v - u)

    v += 0.5 * (0.04 * pow(v,2) + 5 * v + 140 - u + next_input)
    v += 0.5 * (0.04 * pow(v,2) + 5 * v + 140 - u + next_input)

    u = u + a * (b * v - u)
    
    # itr = 0
    # for h in v.get():
    #   if itr < excite:
    #     print str(itr) + '-' * 50
    #     print 0.04 * pow(v.get()[itr],2) + 5 * v.get()[itr] + 140 - u.get()[itr] + next_input.get()[itr]
    #     print 'v: ' + str(v.get()[itr])
    #     print 'u: ' + str(u.get()[itr])
    #     print fired.get()[itr]
    #     print next_input.get()[itr]
    #   itr+=1
    stdp_krnl(dopamine, c, fired, firetimes, weights, post_gpu)



    #DETERMIN MOTOR NEURON OUTPUT
    if last_input != None:
        inc0, inc1 = checkOutput(fired, neural_output)
        if t < last_stim_time + 20: #20 ms window for checking groups 1 vs 0
            class0 += inc0
            class1 += inc1
        elif t == last_stim_time + 21:
            tot = float(class0 + class1)
            if tot != 0:
                cl1p = class0 / tot
                cl2p = class1 / tot
            else:
                cl1p = .5
                cl2p = .5
            #probs_file.write(str(t) + ' \t' + str(cl1p)+ ' \t' + str(cl2p) + '\n')
        reward = False
        if t in dopa_times:
            dopa_times.remove(t)
            if int(last_class) == 0 and cl1p > cl2p:
                reward = True
            if int(last_class) == 1 and cl2p > cl1p:
                reward = True
            print '*' * 30
            print 'Reward Time: ', reward, ' that dopa Injected : class== ', last_class
            print 'Class0: ', cl1p
            print 'Class1: ', cl2p
            input_delivered = False
            last_input = None
        if reward:
            reward_time = t
        ##Update Dopamine levels based on success with output
        if reward_time == t:
            reward = True
        else:
            reward = False
        dopamine += -dopamine / tau_d + DA(t, reward_time, reward)
        dopamine = max(dopamine, 0.0)
        #dopa_file.write(str(t) + ' \t' + str(dopa) + '\n')

    #print sum(fired.get())
    #print v


    # print weights.get()
    # print fired.get()
    #should have the new voltages now

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
