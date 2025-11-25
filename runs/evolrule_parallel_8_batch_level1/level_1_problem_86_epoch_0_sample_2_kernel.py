import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <bool HAS_DW_BIAS, bool HAS_PW_BIAS>
__global__ void fused_convolution_kernel(
    const float* input,
    const float* depthwise_weights,
    const float* pointwise_weights,
    const float* depthwise_bias,
    const float* pointwise_bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int H_out, int W_out, int C_out,
    int kernel_size, int stride, int padding, int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H_out * W_out * C_out) return;
    
    int c_out = idx % C_out;
    int rem = idx / C_out;
    int w = rem % W_out;
    rem /= W_out;
    int h = rem % H_out;
    int n = rem / H_out;
    
    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        float depth_contrib = 0.0f;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = h * stride - padding + kh * dilation;
                int iw = w * stride - padding + kw * dilation;
                if (ih >= 0 && ih < H_in && iw >=0 && iw < W_in) {
                    int in_offset = n * C_in * H_in * W_in +
                        c_in * H_in * W_in +
                        ih * W_in + iw;
                    float val = input[in_offset];
                    int dw_offset = c_in * kernel_size * kernel_size +
                        kh * kernel_size + kw;
                    depth_contrib += val * depthwise_weights[dw_offset];
                }
            }
        }
        
        #if HAS_DW_BIAS
            depth_contrib += depthwise_bias[c_in];
        #endif
        
        int pw_offset = c_out * C_in + c_in;
        float pw_val = pointwise_weights[pw_offset];
        sum += depth_contrib * pw_val;
    }
    
    #if HAS_PW_BIAS
        sum += pointwise_bias[c_out];
    #endif
    
    int out_offset = n * C_out * H_out * W_out +
        c_out * H_out * W_out +
        h * W_out + w;
    output[out_offset] = sum;
}

extern "C" {
torch::Tensor fused_convolution(
    torch::Tensor input,
    torch::Tensor depthwise_weights,
    torch::Tensor pointwise_weights,
    torch::Tensor depthwise_bias,
    torch::Tensor pointwise_bias,
    int kernel_size, int stride, int padding, int dilation,
    bool has_depthwise_bias, bool has_pointwise_bias
) {
    const int threads_per_block = 256;
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = pointwise_weights.size(0);
    
    // Compute output dimensions
    int H_out = (H_in + 2 * padding - dilation * (kernel_size - 1)) / stride + 1;
    int W_out = (W_in + 2 * padding - dilation * (kernel_size - 1)) / stride + 1;
    
    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());
    
    int num_elements = N * H_out * W_out * C_out;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    if (has_depthwise_bias && has_pointwise_bias) {
        fused_convolution_kernel<true, true><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            depthwise_weights.data_ptr<float>(),
            pointwise_weights.data_ptr<float>(),
            depthwise_bias.data_ptr<float>(),
            pointwise_bias.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C_in, H_in, W_in,
            H_out, W_out, C_out,
            kernel_size, stride, padding, dilation
        );
    } else if (has_depthwise_bias) {
        fused_convolution_kernel<true, false><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            depthwise_weights.data_ptr<float>(),
            pointwise_weights.data_ptr<float>(),
            depthwise_bias.data_ptr<float>(),
            nullptr,
            output.data_ptr<float>(),
            N, C_in, H_in, W_in,
            H_out, W_out, C_out,
            kernel_size, stride, padding, dilation
        );
    } else if (has_pointwise_bias) {
        fused_convolution_kernel<false, true><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            depthwise_weights.data_ptr<float>(),
            pointwise_weights.data_ptr<float>(),
            nullptr,
            pointwise_bias.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C_in, H_in, W_in,
            H_out, W_out, C_out,
            kernel_size, stride, padding, dilation
        );
    } else {
        fused_convolution_kernel<false, false><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            depthwise_weights.data_ptr<float>(),
            pointwise_weights.data_ptr<float>(),
            nullptr,
            nullptr,
            output.data_ptr<float>(),
            N, C_in, H_in, W_in,
            H_out, W_out, C_out,
            kernel_size, stride, padding, dilation
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in fused_convolution: " + std::string(cudaGetErrorString(err)));
    }
    
    return output;
}
}
"""

fused_conv_mod = load_inline(
    name="fused_conv",
    cpp_sources="",
    cuda_sources=cuda_source,
    functions=["fused_convolution"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14", "--expt-relaxed-constexpr"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        depthwise_weights = self.depthwise.weight
        pointwise_weights = self.pointwise.weight
        depthwise_bias = self.depthwise.bias
        pointwise_bias = self.pointwise.bias

        has_depthwise_bias = depthwise_bias is not None
        has_pointwise_bias = pointwise_bias is not None

        # Handle cases where bias tensors are None by passing empty tensors
        depthwise_bias = depthwise_bias if has_depthwise_bias else torch.empty(0, device=x.device)
        pointwise_bias = pointwise_bias if has_pointwise_bias else torch.empty(0, device=x.device)

        return fused_conv_mod.fused_convolution(
            x,
            depthwise_weights,
            pointwise_weights,
            depthwise_bias,
            pointwise_bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            has_depthwise_bias,
            has_pointwise_bias,
        )