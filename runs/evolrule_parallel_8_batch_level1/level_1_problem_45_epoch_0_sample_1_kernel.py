import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void avg_pool_kernel(
    const T* input,
    T* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int divisor_override
) {
    int oh = blockIdx.y;
    int ow = blockIdx.x;
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    if (idx >= batch_size * channels) {
        return;
    }

    int b = idx / channels;
    int c = idx % channels;

    int input_h_start = oh * stride - padding;
    int input_w_start = ow * stride - padding;
    int input_h_end = input_h_start + kernel_size;
    int input_w_end = input_w_start + kernel_size;

    input_h_start = max(input_h_start, 0);
    input_w_start = max(input_w_start, 0);
    input_h_end = min(input_h_end, input_height);
    input_w_end = min(input_w_end, input_width);

    int valid_h = input_h_end - input_h_start;
    int valid_w = input_w_end - input_w_start;
    int count = valid_h * valid_w;

    T sum = 0;
    if (count > 0) {
        for (int h = input_h_start; h < input_h_end; ++h) {
            for (int w = input_w_start; w < input_w_end; ++w) {
                int offset = b * channels * input_height * input_width +
                            c * input_height * input_width +
                            h * input_width +
                            w;
                sum += input[offset];
            }
        }
    }

    T output_val = (count > 0) ? sum / static_cast<T>(count) : 0;

    int output_offset = b * channels * output_height * output_width +
                        c * output_height * output_width +
                        oh * output_width +
                        ow;
    output[output_offset] = output_val;
}

void avg_pool_cuda_float(
    torch::Tensor input,
    torch::Tensor output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int divisor_override
) {
    dim3 block(32, 32);
    dim3 grid(output_width, output_height);

    auto stream = at::cuda::getCurrentCUDAStream();

    avg_pool_kernel<float><<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        divisor_override
    );

    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool_cuda", &avg_pool_cuda_float, "Average Pool CUDA Kernel");
}
"""

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

        # Compile the CUDA kernel
        self.avg_pool_cuda = load_inline(
            name="avg_pool_cuda",
            cuda_sources=avg_pool_source,
            functions=["avg_pool_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        output_H = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_W = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = torch.empty(N, C, output_H, output_W, device=x.device, dtype=x.dtype)

        self.avg_pool_cuda(
            x,
            output,
            N, C, H, W,
            output_H, output_W,
            self.kernel_size,
            self.stride,
            self.padding,
            0  # divisor_override
        )

        return output