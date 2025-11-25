def get_inputs():
    x = torch.rand(batch_size, features, sequence_length).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]