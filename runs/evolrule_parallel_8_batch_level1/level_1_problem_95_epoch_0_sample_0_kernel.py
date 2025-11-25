batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda(), torch.randint(0, num_classes, (batch_size,)).cuda()]

def get_init_inputs():
    return []