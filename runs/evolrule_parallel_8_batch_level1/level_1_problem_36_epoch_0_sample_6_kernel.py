import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rmsnorm_kernel = load_inline(
            name='rmsnorm',
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda_runtime.h>
            #include <cub/cub.cuh>

            template <typename T>
            __global__ void rmsnorm_forward_kernel(
                const T* __restrict__ x, T* __restrict__ out, T eps,
                int batch_size, int features, int elements_per_feature) {{
                extern __shared__ T shared[];
                const int tid = threadIdx.x;
                const int width = features * elements_per_feature;
                const int total_elements = batch_size * width;
                const int idx = blockIdx.x * blockDim.x + threadIdx.x;

                T sum = 0;
                if (idx < total_elements) {{
                    int batch = idx / width;
                    int pos = idx % width;
                    int feature = pos / elements_per_feature;
                    T val = x[batch * width + pos];
                    sum = val * val;
                }}
                shared[tid] = sum;
                __syncthreads();

                // Perform block reduction to compute sum of squares per feature
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
                    if (tid < s) {{
                        shared[tid] += shared[tid + s];
                    }}
                    __syncthreads();
                }}

                if (tid == 0) {{
                    T mean_sq = shared[0] / (elements_per_feature * batch_size);
                    T inv_std = rsqrt(mean_sq + eps);
                    for (int b = 0; b < batch_size; ++b) {{
                        for (int f = 0; f < features; ++f) {{
                            int base = b * width + f * elements_per_feature;
                            for (int e = 0; e < elements_per_feature; ++e) {{
                                out[base + e] = x[base + e] * inv_std;
                            }}
                        }}
                    }}
                }}
            }}

            template <typename T>
            torch::Tensor rmsnorm_forward(torch::Tensor x, T eps, int features) {{
                const int batch_size = x.size(0);
                const int elements_per_feature = x.numel() / (batch_size * features);
                const int block_size = 256;
                const int grid_size = (x.numel() + block_size - 1) / block_size;
                auto out = torch::empty_like(x);

                rmsnorm_forward_kernel<T><<<grid_size, block_size, block_size * sizeof(T)>>>(
                    x.data_ptr<T>(), out.data_ptr<T>(), eps,
                    batch_size, features, elements_per_feature
                );
                return out;
            }}

            at::Tensor rmsnorm_forward_cuda(at::Tensor x, at::Scalar eps, int64_t features) {{
                AT_ASSERTM(x.device().is_cuda(), "x must be a CUDA tensor");
                auto eps_ = eps.to<float>();
                if (x.scalar_type() == at::ScalarType::Float) {{
                    return rmsnorm_forward<float>(x, eps_, features);
                }} else if (x.scalar_type() == at::ScalarType::Half) {{
                    return rmsnorm_forward<__half>(x, eps_, features);
                }} else {{
                    AT_ERROR("Unsupported data type");
                }}
            }}
            """,
            cpp_sources="""
            at::Tensor rmsnorm_forward_cuda(at::Tensor x, at::Scalar eps, int64_t features);
            """,
            functions=["rmsnorm_forward_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm_kernel.rmsnorm_forward_cuda(x, self.eps, self.num_features)

# Ensure the input generation matches the original's dimensions
def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()  # Move to CUDA
    return [x]

def get_init_inputs():
    return [features]  # The original's get_init_inputs returns features, so keep it