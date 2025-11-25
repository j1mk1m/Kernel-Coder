import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void combine_clamp_mult(const float* x, const float* m, float* o,
                                  int B, int C, int D, int H, int W,
                                  float min, float max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B*C*D*H*W) return;
    int w = idx % W; idx /= W;
    int h = idx % H; idx /= H;
    int d = idx % D; idx /= D;
    int c = idx % C; int b = idx/C;
    float v = x[b*C*D*H*W + c*D*H*W + d*H*W + h*W + w];
    v = fmaxf(fminf(v, max), min);
    v *= m[c];
    o[b*C*D*H*W + c*D*H*W + d*H*W + h*W + w] = v;
}

__global__ void max_dim1(const float* x, float* o,
                        int B, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B*D*H*W) return;
    int w = idx % W; idx /= W;
    int h = idx % H; idx /= H;
    int d = idx % D; int b = idx/D;
    float max_v = -std::numeric_limits<float>::infinity();
    for(int c=0;c<C;++c) {
        int off = b*C*D*H*W + c*D*H*W + d*H*W + h*W + w;
        float v = x[off];
        if(v > max_v) max_v = v;
    }
    o[idx] = max_v;
}

torch::Tensor combine_clamp_mult_cuda(torch::Tensor x, torch::Tensor m,
                                      float min, float max) {
    auto B = x.size(0), C = x.size(1), D = x.size(2),
         H = x.size(3), W = x.size(4);
    auto out = torch::empty_like(x);
    int num = B*C*D*H*W;
    dim3 blocks((num + 255)/256,1,1);
    dim3 threads(256,1,1);
    combine_clamp_mult<<<blocks, threads>>>(
        x.data_ptr<float>(), m.data_ptr<float>(), out.data_ptr<float>(),
        B,C,D,H,W, min, max);
    return out;
}

torch::Tensor max_dim1_cuda(torch::Tensor x) {
    auto B = x.size(0), C = x.size(1), D = x.size(2),
         H = x.size(3), W = x.size(4);
    auto out = torch::empty({B,D,H,W}, x.options());
    int num = B*D*H*W;
    dim3 blocks((num + 255)/256,1,1);
    dim3 threads(256,1,1);
    max_dim1<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                 B,C,D,H,W);
    return out;
}
"""

cpp_sources = """
torch::Tensor combine_clamp_mult_cuda(torch::Tensor, torch::Tensor, float, float);
torch::Tensor max_dim1_cuda(torch::Tensor);
"""

mods = load_inline(
    name="ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["combine_clamp_mult_cuda", "max_dim1_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fuse = mods.combine_clamp_mult_cuda
        self.max = mods.max_dim1_cuda

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.instance_norm(x)
        x = self.fuse(x, self.multiplier, self.clamp_min, self.clamp_max)
        x = self.max(x)
        return x