# The original get_inputs() function from the problem statement needs to be adjusted to use CUDA tensors
def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]