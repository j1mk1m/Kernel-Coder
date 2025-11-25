import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Load the fused CUDA kernel
        fused_ops_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void fused_operations_kernel(
            const float* conv_out, const float* scaling_factor, const float* bias,
            float* out, int batch_size, int out_channels, int depth, int height, int width) 
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * out_channels * depth * height * width) return;

            int w = idx % width;
            int h = (idx / width) % height;
            int d = (idx / (width * height)) % depth;
            int c = (idx / (width * height * depth)) % out_channels;
            int n = idx / (out_channels * depth * height * width);

            float val = conv_out[idx];
            val *= scaling_factor[c];  // scaling_factor is (out_channels, 1, 1, 1)
            val = tanh(val);
            val *= bias[c];  // bias is (out_channels, 1, 1, 1)
            out[idx] = 1.0 / (1.0 + exp(-val));  // sigmoid
        }

        torch::Tensor fused_operations_cuda(
            torch::Tensor conv_out, torch::Tensor scaling_factor, torch::Tensor bias) 
        {
            const int batch_size = conv_out.size(0);
            const int out_channels = conv_out.size(1);
            const int depth = conv_out.size(2);
            const int height = conv_out.size(3);
            const int width = conv_out.size(4);

            auto out = torch::empty_like(conv_out);
            const int total_elements = batch_size * out_channels * depth * height * width;
            const int block_size = 256;
            const int num_blocks = (total_elements + block_size - 1) / block_size;

            fused_operations_kernel<<<num_blocks, block_size>>>(
                conv_out.data_ptr<float>(), scaling_factor.data_ptr<float>(), 
                bias.data_ptr<float>(), out.data_ptr<float>(), 
                batch_size, out_channels, depth, height, width
            );

            return out;
        }
        """

        fused_ops_cpp = "torch::Tensor fused_operations_cuda(torch::Tensor, torch::Tensor, torch::Tensor);"
        self.fused_ops = load_inline(
            name="fused_ops",
            cpp_sources=fused_ops_cpp,
            cuda_sources=fused_ops_source,
            functions=["fused_operations_cuda"],
            verbose=False
        )

    def forward(self, x):
        x = self.conv(x)
        # Since scaling_factor and bias are parameters, they need to be detached if required
        # But in PyTorch, parameters are automatically tracked, so no need for detach here
        x = self.fused_ops.fused_operations_cuda(x, self.scaling_factor, self.bias)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 3
    depth, height, width = 16, 64, 64
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    scaling_factor = 2
    bias_shape = (out_channels, 1, 1, 1)
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]