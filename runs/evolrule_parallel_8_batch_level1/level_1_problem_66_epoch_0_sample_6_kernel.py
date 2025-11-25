import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename T>
__global__ void im2col_gpu_kernel(
    const T* data_im, 
    const int channels,
    const int depth, const int height, const int width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    T* data_col) {

    int output_depth = (depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1;
    int output_height = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int output_width = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    const int num_kernels = output_depth * output_height * output_width;
    const int channel_size = kernel_d * kernel_h * kernel_w;
    const int num_channels = channels;

    CUDA_1D_KERNEL_LOOP(index, channels * channel_size * num_kernels) {
        int c_col = index % channel_size;
        int c_im = (index / channel_size) % channels;
        int item = index / (channel_size * channels);

        int w_out = item % output_width;
        int h_out = (item / output_width) % output_height;
        int d_out = item / (output_width * output_height);

        int d_in = d_out * stride_d - pad_d;
        int h_in = h_out * stride_h - pad_h;
        int w_in = w_out * stride_w - pad_w;

        int k_d = c_col / (kernel_h * kernel_w);
        int k_rem = c_col % (kernel_h * kernel_w);
        int k_h = k_rem / kernel_w;
        int k_w = k_rem % kernel_w;

        d_in += k_d * dilation_d;
        h_in += k_h * dilation_h;
        w_in += k_w * dilation_w;

        if (d_in < 0 || d_in >= depth || h_in < 0 || h_in >= height || w_in < 0 || w_in >= width) {
            data_col[index] = 0;
        } else {
            int offset = c_im * depth * height * width + d_in * height * width + h_in * width + w_in;
            data_col[index] = data_im[offset];
        }
    }
}

torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);

    int out_depth = (in_depth + 2 * padding_d - dilation_d * (kernel_depth - 1) - 1) / stride_d + 1;
    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

    int channels_col = in_channels * kernel_depth * kernel_height * kernel_width;
    int height_col = out_depth * out_height * out_width;

    auto options = input.options();
    torch::Tensor data_col = torch::empty({batch_size, channels_col, height_col}, options);

    dim3 threads(256);
    dim3 blocks((channels_col * height_col + threads.x - 1) / threads.x);

    im2col_gpu_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        in_channels,
        in_depth, in_height, in_width,
        kernel_depth, kernel_height, kernel_width,
        padding_d, padding_h, padding_w,
        stride_d, stride_h, stride_w,
        dilation_d, dilation_h, dilation_w,
        data_col.data_ptr<float>());

    auto weight_reshaped = weight.view({out_channels, channels_col});

    auto output = torch::empty({batch_size, out_channels, height_col}, options);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0;
    const float beta = 0.0;

    int m = out_channels;
    int n = height_col;
    int k = channels_col;

    int lda = k;
    int ldb = n;
    int ldc = n;

    int stride_a = m * k * sizeof(float);
    int stride_b = k * n * sizeof(float);
    int stride_c = m * n * sizeof(float);

    auto data_col_batched = data_col.permute({0, 2, 1}).contiguous();
    auto data_col_batched_ptr = data_col_batched.data_ptr<float>();
    auto weight_ptr = weight_reshaped.data_ptr<float>();
    auto output_ptr = output.data_ptr<float>();

    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        weight_ptr, CUDA_R_32F, lda,
        data_col_batched_ptr, CUDA_R_32F, ldb,
        &beta,
        output_ptr, CUDA_R_32F, ldc,
        batch_size,
        stride_a, stride_b, stride_c);

    cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("CUBLAS error");
    }

    output = output.view({batch_size, out_channels, out_depth, out_height, out_width});

    if (bias.defined()) {
        output += bias.view({1, -1, 1, 1, 1});
    }

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups);
"""

conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        self.bias_param = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x):
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        dilation_d, dilation_h, dilation_w = self.dilation

        bias = self.bias_param if self.bias else torch.empty(0)

        output = conv3d.conv3d_forward(
            x,
            self.weight,
            bias,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            dilation_d, dilation_h, dilation_w,
            self.groups
        )

        return output