# The original get_inputs and batch_size/dim definitions remain unchanged in the problem's context
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []