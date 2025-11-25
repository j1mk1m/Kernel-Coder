import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Define and load the custom CUDA kernel for logsumexp + ReLU
        logsumexp_relu_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        #define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
        #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

        __global__ void logsumexp_relu_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            int batch_size,
            int channels,
            int depth,
            int height,
            int width
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * depth * height * width)
                return;

            int b = idx / (depth * height * width);
            int pos = idx % (depth * height * width);
            int d = pos / (height * width);
            int hw = pos % (height * width);
            int h = hw / width;
            int w = hw % width;

            float sum = 0.0f;
            const int input_stride = depth * height * width;
            for (int c = 0; c < channels; ++c) {
                int input_offset = b * channels * input_stride + c * input_stride +
                                   d * height * width + h * width + w;
                sum += expf(input[input_offset]);
            }

            float logsumexp_val = logf(sum);
            output[idx] = (logsumexp_val > 0) ? logsumexp_val : 0.0f;
        }

        torch::Tensor logsumexp_relu_cuda(torch::Tensor input) {
            CHECK_CUDA(input);
            CHECK_CONTIGUOUS(input);

            auto output = torch::empty(
                {input.size(0), 1, input.size(2), input.size(3), input.size(4)},
                input.options()
            );

            int batch_size = input.size(0);
            int channels = input.size(1);
            int depth = input.size(2);
            int height = input.size(3);
            int width = input.size(4);

            int num_elements = batch_size * depth * height * width;
            dim3 block(256);
            dim3 grid((num_elements + block.x - 1) / block.x);

            logsumexp_relu_kernel<<<grid, block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                channels,
                depth,
                height,
                width
            );

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            }

            return output;
        }
        """

        logsumexp_relu_cpp_source = (
            "torch::Tensor logsumexp_relu_cuda(torch::Tensor input);"
        )

        # Compile the kernel
        self.logsumexp_relu = load_inline(
            name="logsumexp_relu",
            cpp_sources=logsumexp_relu_cpp_source,  # Fixed variable name
            cuda_sources=logsumexp_relu_source,
            functions=["logsumexp_relu_cuda"],
            verbose=True,
            extra_cflags=["-O3"],
            extra_ldflags=[""]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        # Apply custom kernel for logsumexp and ReLU
        x = self.logsumexp_relu.logsumexp_relu_cuda(x.contiguous())
        return x

# The get_inputs and get_init_inputs functions remain the same as in the original code