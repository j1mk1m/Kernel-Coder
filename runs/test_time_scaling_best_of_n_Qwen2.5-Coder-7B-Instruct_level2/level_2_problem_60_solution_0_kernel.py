def get_inputs():
    batch_size = 128
    in_channels = 3
    out_channels = 16
    depth, height, width = 16, 32, 32
    kernel_size = 3
    stride = 2
    padding = 1
    groups = 4
    eps = 1e-5
    return [torch.rand(batch_size, in_channels, depth, height, width)]