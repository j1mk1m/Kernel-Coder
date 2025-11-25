import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

        # Define the custom CUDA kernel for scaling and min operation
        scale_min_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <limits>

        template <typename scalar_t>
        __global__ void scale_min_kernel(const scalar_t* input, scalar_t* output,
                                        int batch_size, int out_channels,
                                        int height, int width, scalar_t scale) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * height * width)
                return;

            int batch = idx / (height * width);
            int rem = idx % (height * width);
            int x = rem / width;
            int y = rem % width;

            scalar_t min_val = std::numeric_limits<scalar_t>::max();

            for (int c = 0; c < out_channels; ++c) {
                int input_pos = batch * out_channels * height * width
                               + c * height * width
                               + x * width + y;
                scalar_t val = input[input_pos] * scale;
                if (val < min_val) {
                    min_val = val;
                }
            }

            int output_pos = batch * height * width + x * width + y;
            output[output_pos] = min_val;
        }

        torch::Tensor scale_min_cuda(torch::Tensor input, float scale) {
            const auto batch_size = input.size(0);
            const auto out_channels = input.size(1);
            const auto height = input.size(2);
            const auto width = input.size(3);

            auto output = torch::empty({batch_size, 1, height, width},
                                      device=input.device(), dtype=input.dtype());

            const int elements = batch_size * height * width;
            const int threads = 256;
            const int blocks = (elements + threads - 1) / threads;

            const auto stream = at::cuda::getCurrentCUDAStream();
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "scale_min_cuda", ([&] {
                auto scale_val = static_cast<scalar_t>(scale);
                scale_min_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    input.data<scalar_t>(),
                    output.data<scalar_t>(),
                    batch_size,
                    out_channels,
                    height,
                    width,
                    scale_val
                );
            }));

            return output;
        }
        """
        scale_min_cpp = "torch::Tensor scale_min_cuda(torch::Tensor input, float scale);"

        # Compile the kernel
        self.scale_min = load_inline(
            name="scale_min",
            cpp_sources=scale_min_cpp,
            cuda_sources=scale_min_source,
            functions=["scale_min_cuda"],
            verbose=True
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.scale_min.scale_min_cuda(x, self.scale_factor)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]