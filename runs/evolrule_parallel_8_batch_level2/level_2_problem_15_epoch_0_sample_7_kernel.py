import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        
        # Inline CUDA kernel for BatchNorm + Mean Subtraction fusion
        self.fused_kernel = load_inline(
            name="fused_norm_subtract",
            cpp_sources=f"""
                torch::Tensor fused_norm_subtract(torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var, 
                                                  torch::Tensor weight, torch::Tensor bias, float eps);
            """,
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <ATen/cuda/CUDAContext.h>

                __global__ void fused_batch_norm_subtract_kernel(
                    const float* x_data, float* y_data, 
                    const float* running_mean, const float* running_var,
                    const float* weight, const float* bias, 
                    float eps, int N, int C, int D, int H, int W
                ) {{
                    int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index >= N) return;

                    int c = index % C;
                    int n = index / (C * D * H * W);
                    int d = (index / (H * W)) % D;
                    int h = (index / W) % H;
                    int w = index % W;

                    float x_val = x_data[index];
                    float mean = running_mean[c];
                    float var = running_var[c];
                    float inv_std = 1.0f / sqrt(var + eps);
                    float normed = (x_val - mean) * inv_std;

                    if (weight != nullptr) normed *= weight[c];
                    if (bias != nullptr) normed += bias[c];

                    // Compute mean over spatial dimensions (D, H, W)
                    extern __shared__ float shared[];
                    int tid = threadIdx.x;
                    shared[tid] = normed;
                    __syncthreads();

                    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {{
                        if (tid < stride) {{
                            shared[tid] += shared[tid + stride];
                        }}
                        __syncthreads();
                    }}

                    float mean_val = (tid == 0) ? shared[0]/(D*H*W) : 0.0f;
                    __syncthreads();
                    mean_val = blockBroadcast<float>(mean_val);

                    y_data[index] = normed - mean_val;
                }}

                template<typename T>
                __device__ T blockBroadcast(T val) {{
                    T result;
                    if (threadIdx.x == 0) result = val;
                    __syncthreads();
                    return result;
                }}

                torch::Tensor fused_norm_subtract(
                    torch::Tensor x,
                    torch::Tensor running_mean,
                    torch::Tensor running_var,
                    torch::Tensor weight,
                    torch::Tensor bias,
                    float eps
                ) {{
                    const int C = x.size(1);
                    const int N = x.numel() / C;
                    const int D = x.size(2);
                    const int H = x.size(3);
                    const int W = x.size(4);

                    auto y = torch::empty_like(x);
                    const int threads = 256;
                    const int blocks = (x.numel() + threads - 1) / threads;

                    AT_DISPATCH_FLOATING_TYPES(x.type(), "fused_norm_subtract", ([&] {{
                        auto x_data = x.data_ptr<scalar_t>();
                        auto y_data = y.data_ptr<scalar_t>();
                        auto mean = running_mean.data_ptr<scalar_t>();
                        auto var = running_var.data_ptr<scalar_t>();
                        auto w = (weight.defined()) ? weight.data_ptr<scalar_t>() : nullptr;
                        auto b = (bias.defined()) ? bias.data_ptr<scalar_t>() : nullptr;

                        fused_batch_norm_subtract_kernel<<<blocks, threads, threads * sizeof(float)>>>(
                            x_data, y_data,
                            mean, var,
                            w, b,
                            eps, N, C, D, H, W
                        );
                    }}));

                    cudaDeviceSynchronize();
                    return y;
                }}
            """,
            functions=["fused_norm_subtract"],
            verbose=False
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        # Extract batch norm parameters
        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var
        weight = self.batch_norm.weight
        bias = self.batch_norm.bias
        eps = self.batch_norm.eps

        # Apply fused batch norm + mean subtraction
        x = self.fused_kernel.fused_norm_subtract(
            x,
            running_mean.contiguous(),
            running_var.contiguous(),
            weight.contiguous() if weight is not None else torch.tensor([]),
            bias.contiguous() if bias is not None else torch.tensor([]),
            eps
        )
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]