#!/usr/bin/python
# Import all the necessaries
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.driver import device_attribute as A
from DemoMetaMatrixmulCheetah import matrixmul_opt
from pycuda.elementwise import ElementwiseKernel
import pycuda.autoinit

import argparse
import numpy
import sys



#### PARSE ARGS #### 
"""
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
"""
num_neurons = 1000
sim_length = 100
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

# Synapses per neuron
synapses_per = 100
try:
    weights = numpy.ones((num_neurons, num_neurons)).astype(numpy.float32)
except ValueError:
    sys.stderr.write("There is not enough memory for %i neurons. Try again with fewer neurons.\n"%(num_neurons))
c = numpy.ones((num_neurons, num_neurons)).astype(numpy.uint32)

# fire events - these are boolean but use floats to avoid type conflicts
fired = numpy.zeros((1, num_neurons)).astype(numpy.float32)
# fire time
fire_time = numpy.ones((num_neurons, 1)).astype(numpy.uint32)
# for each neuron, indices of postsynaptic neurons
post = numpy.zeros((num_neurons, synapses_per)).astype(numpy.float32)
pe_left = [synapses_per]*num_neurons
# DEBUG: Randomly populate post
# Generate list of postsynaptic connections
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
post = numpy.array(post).astype(numpy.uint32)

#threadid - provided by CUDA

#dopamine
dopamine = 0

# Time constant for dopamine
tau_c = 500.0
eligibility_trace = -1
#### /INITIALIZATION ####

#### STDP KERNEL ####
stdp_krnl = ElementwiseKernel(
    "float dopamine, float *c, int *fired, int *fire_time,"+
      "float *weights, int *post",
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
#### /STDP KERNEL ####

#### DATA TRANSFER ####
# TODO: This should be split up so that we don't fill up our memory.
weights_gpu = gpuarray.to_gpu(weights)
c_gpu = gpuarray.to_gpu(c)
fired_gpu = gpuarray.to_gpu(fired)
fire_time_gpu = gpuarray.to_gpu(fire_time)
post_gpu = gpuarray.to_gpu(post)
#### /DATA TRANSFER ####

stdp_krnl(dopamine, c_gpu, fired_gpu, fire_time_gpu, weights_gpu, post_gpu)


#### MAIN LOOP ####
for i in xrange(sim_length):
    # Calculate input
    # Input for a neuron is the sum of the synaptic weights of all neurons that
    # fired onto it
    inputs = matrixmul_opt(fired, weights)[0][0]
    
    # Do firing

    # Do STDP
    stdp_krnl(dopamine, c_gpu, fired_gpu, fire_time_gpu, weights_gpu, post_gpu)


#### /MAIN LOOP ####    

for neu,inp in enumerate(inputs):
    if inp>0:
        print "Neuron %i has an input of %i."%(neu,inp)



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
