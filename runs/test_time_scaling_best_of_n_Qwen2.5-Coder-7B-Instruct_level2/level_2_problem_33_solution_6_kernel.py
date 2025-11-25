import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

model_new_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void model_new_kernel(const float* x, const float* weight, const float* bias,