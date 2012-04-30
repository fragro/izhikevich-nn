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


kernel_code_template = """
  __global__ void firetime_init(float *a, float *b)
  {
    int idx = threadIdx.x;
    if(a[idx] > 30){
      b[idx] = 1;
    }
  };

  __global__ void firetime_iter(float *a, float *b, float c)
  {
    int idx = threadIdx.x;
    if(a[idx] == 1){
      b[idx] = c;
    }
  }
__global__ void MatrixMulKernel(float *A, float *B, float *C)
{

  const uint wA = %(MATRIX_SIZE)s;
  const uint wB = %(MATRIX_SIZE)s;  
  
  // Block index
  const uint bx = blockIdx.x;
  const uint by = blockIdx.y;

  // Thread index
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  const uint aBegin = wA * %(BLOCK_SIZE)s * by;
  // Index of the last sub-matrix of A processed by the block
  const uint aEnd = aBegin + wA - 1;
  // Step size used to iterate through the sub-matrices of A
  const uint aStep = %(BLOCK_SIZE)s;

  // Index of the first sub-matrix of B processed by the block
  const uint bBegin = %(BLOCK_SIZE)s * bx;
  // Step size used to iterate through the sub-matrices of B
  const uint bStep = %(BLOCK_SIZE)s * wB;

  // The element of the block sub-matrix that is computed
  // by the thread
  float Csub = 0;
  // Loop over all the sub-matrices of A and B required to
  // compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) 
    {
      // Shared memory for the sub-matrix of A
      __shared__ float As[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
      // Shared memory for the sub-matrix of B
      __shared__ float Bs[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];

      // Load the matrices from global memory to shared memory;
      // each thread loads one element of each matrix
      As[ty][tx] = A[a + wA * ty + tx];
      Bs[ty][tx] = B[b + wB * ty + tx];
      // Synchronize to make sure the matrices are loaded
      __syncthreads();

      // Multiply the two matrices together;
      // each thread computes one element
      // of the block sub-matrix
      for (int k = 0; k < %(BLOCK_SIZE)s; ++k)
    Csub += As[ty][k] * Bs[k][tx];

      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      __syncthreads();
    }

  // Write the block sub-matrix to global memory;
  // each thread writes one element
  const uint c = wB * %(BLOCK_SIZE)s * by + %(BLOCK_SIZE)s * bx;
  C[c + wB * ty + tx] = Csub;
}
"""

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
# define the (square) matrix size
MATRIX_SIZE = 4

# define size of blocks and tiles sub-matrix 
# (we assume that the block size is same as tile size)
TILE_SIZE = 4
BLOCK_SIZE = TILE_SIZE

# get the kernel code from the template 
# by specifying the constants MATRIX_SIZE and BLOCK_SIZE
kernel_code = kernel_code_template % { 
    'MATRIX_SIZE': num_neurons,
    'BLOCK_SIZE': BLOCK_SIZE,
    }

# compile the kernel code
mod = pycuda.compiler.SourceModule(kernel_code)

#dopamine
synapses_per = 100
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
# Set up necessary arrays
# Internal neuron variables. Synaptic weights and c.
# in weights, rows are presyn & cols are psotsyn
try:
    weights_cpu = numpy.array([[rand.random() for k in xrange(num_neurons)]  for h in xrange(num_neurons)]).astype(numpy.float32)
    weights = gpuarray.to_gpu(weights_cpu)
except ValueError:
    sys.stderr.write("There is not enough memory for %i neurons. Try again with fewer neurons.\n"%(num_neurons))

# fire events - these are boolean but use floats to avoid type conflicts
fired_cpu = numpy.zeros((1, num_neurons)).astype(numpy.float32)

#TESTING
fired_cpu = numpy.array(map(lambda x: int(x>=.6) , [rand.random() for i in xrange(num_neurons)])).astype(numpy.float32)
fired = gpuarray.to_gpu(fired_cpu)

inputs = gpuarray.zeros((num_neurons,num_neurons), numpy.float32)

# fire time
fire_time = numpy.ones((1, num_neurons)).astype(numpy.uint32)
#threadid - provided by CUDA

input_delivered = False

rand_exc = curand(excite, dtype=numpy.float32)
rand_inh = curand(inhib, dtype=numpy.float32)
firetimes = gpuarray.zeros(excite+inhib, dtype=numpy.float32)

v = -65 * numpy.ones(excite+inhib)
u = v.copy()
v = gpuarray.to_gpu(numpy.array(v))

a = [0.02 for i in xrange(excite)]
a.extend([0.02+0.08*rand.random() for i in xrange(inhib)])
a = gpuarray.to_gpu(numpy.array(a))

#timescale of the recovery variable
b = [0.02 for i in xrange(excite)]
b.extend([0.25-0.05*rand.random() for i in xrange(inhib)])
u *= b[0]
u = gpuarray.to_gpu(u)
b = gpuarray.to_gpu(numpy.array(b))

c = [-65+15*pow(rand.random(),2) for i in xrange(excite)]
c.extend([-65 for i in xrange(inhib)])
c = gpuarray.to_gpu(numpy.array(c))
#sensitivity of the recovery variable


#mV -- after-spike reset of the membrane potential
d = [8-6*pow(rand.random(),2) for i in xrange(excite)]
d.extend([2 for i in xrange(inhib)])
d = gpuarray.to_gpu(numpy.array(d))


#### /INITIALIZATION ####
neural_input, inputset, eligible_neurons, classes = initInput(excite, trainfile, num_neurons_per_input)
#Need some times for stimulus to occur
stimulus_times = range(100,sim_length,stimulus_interval)
pp.pprint(stimulus_times)

####CUDA SPECIFIC
BLOCK_X=4
BLOCK_Y=4
block_size = 16
work_size = 1
###KERNEL FOR DETERMINGING THE INPUT

firing_v_comb = ElementwiseKernel(
        "float *v, float *f, float *c",
        "v[i] = f[i] * c[i] + !f[i] * v[i]",
        "linear_combination")


firing_u_comb = ElementwiseKernel(
        "float *v, float *f, float *u, float *d",
        "u[i] = f[i] * (u[i]+d[i]) + !f[i] * u[i]",
        "linear_combination")

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


# Time constant for dopamine
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

#### MAIN LOOP ####
for t in xrange(sim_length):
    # Calculate input
    # Input for a neuron is the sum of the synaptic weights of all neurons that
    # fired onto it
    #### also need to include input from basic.train
    inp, arr = getInput(i, stimulus_times, neural_input, inputset, excite)
    if inp != None:
      print arr

    ################################################################################
    if inp != None and not input_delivered:
        dopamine_time = t + rand.randint(20,reward_wait_time)
        last_stim_time = t
        last_input = inp
        dopa_times.append(dopamine_time)
        class0_record.append(class0)
        class1_record.append(class1)
        class0 = 0.0
        class1 = 0.0

    # should break this down into a kernel as well
    # determining the input is a bit more complex 
    inhibitory = [5 * rand.random() for i in xrange(inhib)]


    if (arr == None and not input_delivered) or input_delivered:
        excitatory = [5 * rand.random() for i in xrange(excite)]

    elif arr != None and not input_delivered: 
        excitatory = [arr + 5 for i in xrange(excite)]
    excitatory.extend(inhibitory)
    input_vector = gpuarray.to_gpu(numpy.array(excitatory)) 

    #### Input is ready to go, now we need to calculate the total inputs/Izekivich model
    #    numpy.array(map(lambda x: x>=30 , v[0:excite+inhib])) 
    if t == 0:
        #MATRIX INPUT OP\
        func = mod.get_function("firetime_init")
        func(v, fired, block=(BLOCK_X,BLOCK_Y,1))
        #print fired * weights
    #update firetimes
    else:
        func = mod.get_function("firetime_iter")
        #func(fired, firetimes, block=(BLOCK_X,BLOCK_Y,1))

    firing_v_comb(v, fired, c)
    firing_u_comb(v, fired, u, d)

    #inputs = matrixmul_opt(weights, fired)
    #print inputs

    # get the kernel function from the compiled module
    # matrixmul = mod.get_function("MatrixMulKernel")

    # # call the kernel on the card
    # matrixmul(
    #     # inputs
    #     fired, weights, 
    #     # output
    #     inputs, 
    #     # grid of multiple blocks
    #     grid = (num_neurons / TILE_SIZE, num_neurons / TILE_SIZE),
    #     # block of multiple threads
    #     block = (TILE_SIZE, TILE_SIZE, 1), 
    # )
    print (fired * weights)
    print weights_cpu[0][0]
    inputs = (fired * weights) + input_vector
    print inputs
    #print inputs == weights * fired #this is false always... random?

    # print  inputs_cpu - inputs.get()
    #print "L2 norm:", la.norm(inputs_cpu - inputs.get())
    #print weights_cpu * fired_cpu
    #print sum(fired.get(numpy.empty((num_neurons), numpy.float32)))
    #print pp.pprint(sum(inputs.get(numpy.empty((num_neurons,num_neurons), numpy.float32))[4]))
    #print inputs.get(numpy.empty((num_neurons,num_neurons), numpy.float32))[0]
    print v.shape
    print u.shape
    print inputs.shape
    v += 0.04 * pow(v,2) + 5 * v + 140 - u + inputs
    u = u + a * (b * v - u)

    stdp_krnl(dopamine, c, fired, fire_times, weights_gpu, post_gpu)

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
