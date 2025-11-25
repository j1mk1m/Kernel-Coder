import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Optimized LogSoftmax kernel
logsoftmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void logsoftmax_kernel(const float* x, float* y, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        int row = idx / dim;
        int col = idx % dim;
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < dim; ++j) {
            max