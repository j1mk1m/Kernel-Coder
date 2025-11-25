def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [8192, 8192]