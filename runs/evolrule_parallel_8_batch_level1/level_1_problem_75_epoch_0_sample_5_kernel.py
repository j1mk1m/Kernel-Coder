# Additional note: The code above assumes that the input tensors are on the GPU. 
# The get_inputs() function has been adjusted to place inputs on the CUDA device.
# The CUDA kernel uses atomicAdd for accumulation to handle potential index collisions, 
# which is necessary due to the parallel nature of the kernel. The grid and block dimensions 
# are set to handle the output spatial dimensions and channel grouping efficiently.