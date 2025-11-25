def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]