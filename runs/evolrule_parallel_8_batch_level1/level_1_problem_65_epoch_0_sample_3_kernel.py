import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose2d(x)

batch_size = 8
in_channels = 64
out_channels = 64
kernel_size = (3, 7)  # larger asymmetric kernel
width = 512
height = 512

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

# Custom CUDA kernel implementation for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int groups
) {
    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    int out_channels = weight.size(1); // weight shape is (in_channels, out_channels, ...)
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int out_width = (in_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto transposed_weight = weight.permute({1, 0, 2, 3}).contiguous();

    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);

    cudnnTensorDescriptor_t input_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, out_channels, in_height, in_width);

    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, kernel_h, kernel_w);

    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, in_channels, out_height, out_width);

    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    size_t workspace_size;
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle, filter_desc, input_desc, conv_desc, output_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &workspace_size);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    auto output = torch::empty({batch_size, in_channels, out_height, out_width}, device, torch::kFloat32);

    float alpha = 1.0f;
    float beta = 0.0f;

    cudnnConvolutionBackwardData(cudnn_handle, &alpha, filter_desc, transposed_weight.data_ptr<float>(), input_desc, input.data_ptr<float>(), conv_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, workspace, workspace_size, &beta, output_desc, output.data_ptr<float>());

    if (bias.defined()) {
        output = output + bias.view({1, -1, 1, 1});
    }

    if (workspace) {
        cudaFree(workspace);
    }
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}
"""

cpp_sources = """
#include <torch/extension.h>

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int groups
);
"""

conv_transpose2d_cuda = load_inline(
    name="conv_transpose2d_cuda",
    cpp_sources=cpp_sources,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1]).cuda())
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels).cuda())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        x = x.cuda()
        return conv_transpose2d_cuda(
            x,
            self.weight,
            self.bias if hasattr(self, 'bias') else torch.tensor([]).cuda(),
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.output_padding[0],
            self.output_padding[1],
            self.groups
        )