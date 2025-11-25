import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# define your custom CUDA operators here
...

class ModelNew(nn.Module):
    ...