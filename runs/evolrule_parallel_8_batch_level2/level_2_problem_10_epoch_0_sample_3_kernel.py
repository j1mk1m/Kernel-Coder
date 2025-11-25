import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        # Fused kernel for Hardtanh, mean, and tanh operations
        fused_kernel_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void fused_operations_kernel(
            const scalar_t* input,
            scalar_t* output,
            int batch_size,
            int channels,
            int h,
            int w,
            float min_val,
            float max_val
        ) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * channels) return;

            const int channel = idx % channels;
            const int batch = idx / channels;

            scalar_t sum = 0;
            for (int i = 0; i < h; ++i) {
                for (int j = 0; j < w; ++j) {
                    int pos = ((batch * channels + channel) * h + i) * w + j;
                    scalar_t val = input[pos];
                    // Apply Hardtanh
                    val = max(static_cast<scalar_t>(min_val), min(val, static_cast<scalar_t>(max_val)));
                    sum += val;
                }
            }
            // Compute mean and apply tanh
            scalar_t mean = sum / (h * w);
            output[idx] = tanh(mean);
        }

        torch::Tensor fused_operations(torch::Tensor input, float min_val, float max_val) {
            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int h = input.size(2);
            const int w = input.size(3);
            const int output_size = batch_size * channels;

            auto output = torch::empty({batch_size, channels, 1, 1}, input.options());

            const int block_size = 256;
            const int num_blocks = (output_size + block_size - 1) / block_size;

            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_operations", ([&] {
                fused_operations_kernel<scalar_t><<<num_blocks, block_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels, h, w,
                    min_val, max_val);
            }));

            return output;
        }
        """
        
        # Compile the fused CUDA kernel
        fused_ops = load_inline(
            name="fused_operations",
            cpp_sources="",
            cuda_sources=fused_kernel_source,
            functions=["fused_operations"],
            verbose=True,
            with_cuda=True
        )
        self.fused_operations = fused_ops.fused_operations

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        # Apply fused operations
        x = self.fused_operations(x, self.hardtanh_min, self.hardtanh_max)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 64
    height = width = 256
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    in_channels = 64  
    out_channels = 64  
    kernel_size = 3
    stride = 1
    padding = 1
    maxpool_kernel_size = 2
    maxpool_stride = 2
    hardtanh_min = -1.0
    hardtanh_max = 1.0
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]