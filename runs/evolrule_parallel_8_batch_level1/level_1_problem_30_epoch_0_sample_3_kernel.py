# The original code's get_inputs() was generating CPU tensors. The optimized code uses CUDA tensors directly.
# Hence, in the new code, get_inputs() now calls .cuda() to ensure inputs are on the GPU.