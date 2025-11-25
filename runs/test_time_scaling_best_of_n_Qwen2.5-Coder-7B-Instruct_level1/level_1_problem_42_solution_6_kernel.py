import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool2d_kernel(const float* input, float* output, int batch_size, int channels, int height, int width, int kernel_size, int stride, int padding, int dilation) {
    int n = blockIdx.z * blockDim.z + threadIdx.z; // batch index
    int c = blockIdx.y * blockDim.y + threadIdx.y; // channel index
    int h_out = blockIdx.x * blockDim.x + threadIdx.x; // output height index

    if (n >= batch_size || c >= channels || h_out >= (height + padding - kernel_size + 1) / stride) {
        return;
    }

    int w_out = ((h_out * stride) - padding);
    int h_start = ((h_out * stride) - padding);
    int w_start = ((w_out * stride) - padding);

    float max_val = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int h_in = h_start + i;
            int w_in = w_start + j;

            if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                float val = input[n * channels * height * width + c * height * width + h_in * width + w_in];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    output[n * channels * (height + padding - kernel_size + 1) / stride * width + c * (height + padding - kernel_size + 1) / stride * width + h_out * width + w_out] = max_val;
}

torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::zeros({batch_size, channels, (height + padding - kernel_size + 1) / stride, (width + padding - kernel_size + 1) / stride}, input.options());

    const int block_size = 32;
    dim3 grid((height + padding - kernel_size + 1) / stride / block_size, channels / block_size, batch_size);
    dim3 block(block_size, block_size, 1);

    maxpool2d_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width, kernel_size, stride, padding, dilation);

    return output;
}
"""

maxpool2d_cpp_source = (
    "torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code for max pool 2d
maxpool2d = load_inline(
    name="maxpool2d",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_source,
    functions=["maxpool2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.maxpool = maxpool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool.maxpool2d_cuda(x, kernel_size, stride, padding, dilation)


batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

model_new = ModelNew(kernel_size, stride, padding, dilation)
inputs = get_inputs()

output = model_new(inputs[0])
print(output.shape)