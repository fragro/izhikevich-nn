#!/usr/bin/python
# Import all the necessaries
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.driver import device_attribute as A
from DemoMetaMatrixmulCheetah import matrixmul_opt
from pycuda.elementwise import ElementWiseKernel
import pycuda.autoinit

import argparse
import numpy
import sys



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
#threadid - provided by CUDA

#dopamine
dopamine = 0

# Time constant for dopamine
tau_c = 500.0
eligibility_trace = -1
#### /INITIALIZATION ####

#### STDP KERNEL ####
stdp_krnl = ElementWiseKernel(
    "float dopamine, float *c, int fire_time, int *fired, float *weights, int *post, float tau_c",
    """
    #define SYN_PER %(synapses_per)i
    #define NUM_NEU %(num_neurons)i
    #define TAU_C %(tau_c)f

    float syn; 
    int both_fired;
    int tau;
    int cur_po;
    int old_c;
    
    /* for each postsynaptic neuron */
    for(int j = 0; j < SYN_PER; j++) {
      cur_po = post[j*NUM_NEU+i];
      syn = weights[cur_po*NUM_NEU+i];
      /* BEGIN IF */
      both_fired = (fired[i]>0 && fired[cur_po]);
      tau = fire_time[cur_po]-fire_time[i];
      /* n_plus = 0.3, n_minus = 3.0 */
      old_c = c[cur_po*NUM_NEU+i];
      dc = (
        -old_c/TAU_C
        + (tau > 0)*((4-syn)*0.3)*2.7182^(-tau / 10.0)
        + (tau < 0)*((syn)*3.0)*2.7182^(-au / 10.0)
        )*both_fired 
      c[cur_po*NUM_NEU+i] = old_c + dc;
      syn += old_c + dc;
      weights[cur_po*NUM_NEU+i] = (syn > 4 || syn < 0)*syn + (syn > 4)*4;

      /* END IF */
    }

    """%{"synapses_per":synapses_per, "tau_c":tau_c, 
          "num_neurons":num_neurons}
) 
#### /STDP KERNEL ####

#### MAIN LOOP ####
for i in xrange(sim_length):
    # Calculate input
    # Input for a neuron is the sum of the synaptic weights of all neurons that
    # fired onto it
    inputs = matrixmul_opt(fired, weights)[0][0]

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
