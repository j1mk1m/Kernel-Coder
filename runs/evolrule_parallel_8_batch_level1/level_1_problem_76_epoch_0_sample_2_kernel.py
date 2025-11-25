import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int dilation,
    int input_length,
    int output_length
) {
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int tid = threadIdx.x;

    // Shared memory for current weights (for oc)
    extern __shared__ float s_weights[];
    int weight_offset = oc * in_channels * kernel_size;
    for (int idx = threadIdx.x; idx < in_channels * kernel_size; idx += blockDim.x) {
        s_weights[idx] = weight[ weight_offset + idx ];
    }
    __syncthreads();

    // Compute the start and end time steps for this thread
    int step_count = (output_length + blockDim.x - 1) / blockDim.x;
    int start_t = tid * step_count;
    int end_t = min(start_t + step_count, output_length);

    for (int t = start_t; t < end_t; t++) {

        float sum = 0.0f;

        // Precompute the kernel positions
        int pos0 = t * stride;
        for (int k = 0; k < kernel_size; k++) {
            int pos = pos0 + k * dilation;
            // Iterate over input channels
            for (int ic = 0; ic < in_channels; ic++) {
                float val = input[ b * in_channels * input_length + ic * input_length + pos ] *
                            s_weights[ ic * kernel_size + k ];
                sum += val;
            }
        }

        if (bias) {
            sum += bias[oc];
        }

        int output_offset = b * out_channels * output_length + oc * output_length + t;
        output[output_offset] = sum;
    }
}

torch::Tensor conv1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int dilation,
    int kernel_size
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int input_length = input.size(2);

    // Compute output length
    int numerator = input_length - dilation * (kernel_size - 1) - 1;
    int output_length = (numerator / stride) + 1;

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    dim3 threads(256);
    dim3 blocks(batch_size, out_channels);

    int shared_mem_size = in_channels * kernel_size * sizeof(float);

    // Launch the kernel
    conv1d_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        input_length,
        output_length
    );

    cudaDeviceSynchronize();
    return output;
}
"""

conv1d_cpp_source = """
torch::Tensor conv1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int dilation,
    int kernel_size
);
"""

# Compile the CUDA code
conv1d_cuda = load_inline(
    name="conv1d_cuda",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_source,
    functions=["conv1d_cuda"],
    verbose=True
)

class Conv1dFromScratch(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False):
        super(Conv1dFromScratch, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size, in_channels, length = x.size()
        assert in_channels == self.in_channels, "Input channel mismatch"

        # Compute output length
        L_in = length
        numerator = L_in - self.dilation * (self.kernel_size - 1) - 1
        out_length = (numerator // self.stride) + 1

        # Launch the CUDA kernel
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        output = conv1d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            bias_tensor.contiguous(),
            self.stride,
            self.dilation,
            self.kernel_size
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = Conv1dFromScratch(in_channels, out_channels, kernel_size, stride, dilation, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d(x)

def get_init_inputs():
    # Parameters for initialization of ModelNew
    return [in_channels, out_channels, kernel_size, stride, dilation]

def get_inputs():
    x = torch.rand(batch_size, in_channels, length).cuda()
    return [x]