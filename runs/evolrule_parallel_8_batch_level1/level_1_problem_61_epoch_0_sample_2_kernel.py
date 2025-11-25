# Original get_inputs and get_init_inputs remain unchanged
def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias]