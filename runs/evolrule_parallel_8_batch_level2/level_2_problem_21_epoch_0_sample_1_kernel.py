import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

        # Define fused kernel for (bias + scale) * input followed by sigmoid
        fused_kernel_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void fused_ops_kernel(const scalar_t* input, const scalar_t* bias, const scalar_t* scale, scalar_t* output, int channels, int spatial_size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= channels * spatial_size) return;

            int c = idx / spatial_size;
            int s = idx % spatial_size;

            // Compute (input + bias) * scale
            scalar_t val = (input[idx] + bias[c]) * scale[c];
            output[idx] = 1.0 / (1.0 + exp(-val));
        }

        torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale) {
            const int batch = input.size(0);
            const int channels = input.size(1);
            const int spatial_size = input.size(2) * input.size(3);
            const int total_elements = input.numel();

            auto output = torch::empty_like(input);

            const int threads = 256;
            const int blocks = (total_elements + threads - 1) / threads;

            AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_ops_cuda", ([&] {
                fused_ops_kernel<scalar_t><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    scale.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    channels,
                    spatial_size
                );
            }));

            return output;
        }
        """

        fused_kernel_header = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale);"

        # Compile fused kernel
        self.fused_ops = load_inline(
            name="fused_ops",
            cpp_sources=[fused_kernel_header],
            cuda_sources=[fused_kernel_source],
            functions=["fused_ops_cuda"],
            verbose=True
        )

    def forward(self, x):
        x = self.conv(x)
        # Apply fused operations (bias + scale + sigmoid)
        x = self.fused_ops.fused_ops_cuda(x, self.bias, self.scale)
        # Apply group norm
        x = self.group_norm(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]

# Global parameters (from problem statement)
batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)