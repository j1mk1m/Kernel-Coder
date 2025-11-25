import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/extension.h>

torch::Tensor avg_pool3d_forward_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
torch::Tensor avg_pool3d_backward_cuda(
    torch::Tensor grad_output, 
    torch::Tensor input, 
    int kernel_size, 
    int stride, 
    int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool3d_forward", &avg_pool3d_forward_cuda, "3D average pool forward");
    m.def("avg_pool3d_backward", &avg_pool3d_backward_cuda, "3D average pool backward");
}
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool3d_forward_kernel(
    const float* input, float* output,
    int kernel_size, int stride, int padding,
    int N, int C, int D, int H, int W,
    int OD, int OH, int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * OD * OH * OW) return;

    int ow = idx % OW;
    int tmp = idx / OW;
    int oh = tmp % OH;
    tmp = tmp / OH;
    int od = tmp % OD;
    tmp = tmp / OD;
    int c = tmp % C;
    int n = tmp / C;

    int d_start = od * stride - padding;
    int h_start = oh * stride - padding;
    int w_start = ow * stride - padding;

    int d_end = d_start + kernel_size;
    int h_end = h_start + kernel_size;
    int w_end = w_start + kernel_size;

    d_start = max(d_start, 0);
    h_start = max(h_start, 0);
    w_start = max(w_start, 0);

    d_end = min(d_end, D);
    h_end = min(h_end, H);
    w_end = min(w_end, W);

    float sum = 0.0f;
    for (int d = d_start; d < d_end; ++d) {
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int input_offset = 
                    n * C * D * H * W +
                    c * D * H * W +
                    d * H * W +
                    h * W +
                    w;
                sum += input[input_offset];
            }
        }
    }

    int kernel_vol = kernel_size * kernel_size * kernel_size;
    float avg = sum / kernel_vol;

    int output_offset = 
        n * C * OD * OH * OW +
        c * OD * OH * OW +
        od * OH * OW +
        oh * OW +
        ow;
    output[output_offset] = avg;
}

__global__ void avg_pool3d_backward_kernel(
    const float* grad_output, float* grad_input,
    int kernel_size, int stride, int padding,
    int N, int C, int D, int H, int W,
    int OD, int OH, int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * OD * OH * OW) return;

    int ow = idx % OW;
    int tmp = idx / OW;
    int oh = tmp % OH;
    tmp = tmp / OH;
    int od = tmp % OD;
    tmp = tmp / OD;
    int c = tmp % C;
    int n = tmp / C;

    float grad_out_val = grad_output[idx];
    float grad_per_element = grad_out_val / (kernel_size * kernel_size * kernel_size);

    int d_start = od * stride - padding;
    int h_start = oh * stride - padding;
    int w_start = ow * stride - padding;

    int d_end = d_start + kernel_size;
    int h_end = h_start + kernel_size;
    int w_end = w_start + kernel_size;

    d_start = max(d_start, 0);
    h_start = max(h_start, 0);
    w_start = max(w_start, 0);

    d_end = min(d_end, D);
    h_end = min(h_end, H);
    w_end = min(w_end, W);

    for (int d = d_start; d < d_end; ++d) {
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int input_offset = 
                    n * C * D * H * W +
                    c * D * H * W +
                    d * H * W +
                    h * W +
                    w;
                atomicAdd(&grad_input[input_offset], grad_per_element);
            }
        }
    }
}

extern "C" {

torch::Tensor avg_pool3d_forward_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int OD = (D + 2 * padding - kernel_size) / stride + 1;
    int OH = (H + 2 * padding - kernel_size) / stride + 1;
    int OW = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({N, C, OD, OH, OW}, input.options());

    int total_elements = N * C * OD * OH * OW;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    avg_pool3d_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size, stride, padding,
        N, C, D, H, W,
        OD, OH, OW
    );

    return output;
}

torch::Tensor avg_pool3d_backward_cuda(
    torch::Tensor grad_output, 
    torch::Tensor input, 
    int kernel_size, 
    int stride, 
    int padding
) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int OD = (D + 2 * padding - kernel_size) / stride + 1;
    int OH = (H + 2 * padding - kernel_size) / stride + 1;
    int OW = (W + 2 * padding - kernel_size) / stride + 1;

    auto grad_input = torch::zeros_like(input);

    int total_elements = N * C * OD * OH * OW;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    avg_pool3d_backward_kernel<<<num_blocks, block_size>>>(
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        kernel_size, stride, padding,
        N, C, D, H, W,
        OD, OH, OW
    );

    return grad_input;
}

}
"""

# Compile the CUDA extension
avg_pool3d = load_inline(
    name="avg_pool3d",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["avg_pool3d_forward", "avg_pool3d_backward"],
    verbose=True,
)

class AvgPool3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding):
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding

        return avg_pool3d.avg_pool3d_forward(input, kernel_size, stride, padding)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding

        grad_input = avg_pool3d.avg_pool3d_backward(grad_output, input, kernel_size, stride, padding)

        return grad_input, None, None, None

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        return AvgPool3dFunction.apply(x, self.kernel_size, self.stride, self.padding)

# The given functions remain unchanged:
batch_size = 16
channels = 32
depth = 128
height = 128
width = 256
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    x = torch.rand(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]