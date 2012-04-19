#!/usr/bin/python
# Import all the necessaries
import pycuda.driver as cuda
from pycuda import gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule

import argparse
import numpy
"""
parser = argparse.ArgumentParser(description="Simulate a network of neurons "+
    "with small-world connectivity.")
parser.add_argument('num_neurons', metavar='neurons', 
    help='the number of neurons to simulate')
parser.add_argument('num_synapses', metavar='synapses',
    help='the number of presynaptic and postsynaptic connections per neuron')
parser.add_argument('-r', '--random', action='store_true',
    help='generate a randomly connected network')

args = parser.parse_args()
num_neurons  = args.num_neurons
num_synapses = args.num_synapses  
"""
num_neurons  = 5
num_synapses = 5
# Set up necessary arrays
# Internal neuron variables. Synaptic weights and c.
weights = numpy.zeros((num_neurons, num_synapses)).astype(numpy.float32)
c       = numpy.zeros((num_neurons, 1)).astype(numpy.float32)
# pn_gpu = gpuarray.to_gpu(per_neuron)

# Alloc memory on GPU
weights_gpu = cuda.mem_alloc(weights.nbytes)
# Transfer data to GPU
cuda.memcpy_htod(weights_gpu, weights)

# Alloc memory on GPU
c_gpu = cuda.mem_alloc(c.nbytes)
# Transfer data to GPU
cuda.memcpy_htod(c_gpu, c)

# Set up a source module
mod = SourceModule("""
    __global__ void update_weights(float *weights, float *c)
    {
       return; 
    }
""")

func = mod.get_function('update_weights')
func(weights_gpu, c_gpu, block=(4,4,1))
