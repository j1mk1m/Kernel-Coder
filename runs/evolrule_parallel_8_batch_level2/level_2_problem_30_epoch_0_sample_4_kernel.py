import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

        # Define the CUDA kernel with proper header declaration
        group_norm_and_hardtanh_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <cmath>

        __global__ void group_norm_and_hardtanh_kernel(
            float* x,
            const int batch_size,
            const int channels,
            const int groups,
            const float eps,
            const float min_val,
            const float max_val
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * channels) return;

            int group_size = channels / groups;
            int group_id = (idx / group_size) % groups;
            int channel_in_group = idx % group_size;

            extern __shared__ float shared[];
            float* means = shared;
            float* vars = shared + groups;

            // Compute mean and variance for each group
            float sum = 0.0;
            for (int i = 0; i < group_size; ++i) {
                sum += x[idx - channel_in_group + i];
            }
            __syncthreads();
            means[group_id] = sum / group_size;

            float var_sum = 0.0;
            for (int i = 0; i < group_size; ++i) {
                float resid = x[idx - channel_in_group + i] - means[group_id];
                var_sum += resid * resid;
            }
            __syncthreads();
            vars[group_id] = var_sum / group_size;

            // Normalize and apply activation
            float mean = means[group_id];
            float var = vars[group_id];
            float std = sqrtf(var + eps);
            float normalized = (x[idx] - mean) / std;
            x[idx] = normalized < min_val ? min_val : (normalized > max_val ? max_val : normalized);
        }

        torch::Tensor group_norm_and_hardtanh_cuda(
            torch::Tensor x,
            int groups,
            float eps,
            float min_val,
            float max_val
        ) {
            const auto batch_size = x.size(0);
            const auto channels = x.size(1);
            const int threads = 256;
            const int elements = batch_size * channels;
            const int blocks = (elements + threads - 1) / threads;
            const int shared_size = groups * 2 * sizeof(float);

            auto x_data = x.contiguous();
            auto out = x_data.clone();

            group_norm_and_hardtanh_kernel<<<blocks, threads, shared_size>>>(
                out.data_ptr<float>(),
                batch_size,
                channels,
                groups,
                eps,
                min_val,
                max_val
            );

            return out;
        }
        """

        # Provide proper C++ header declaration
        cpp_sources = """
        #include <torch/extension.h>
        torch::Tensor group_norm_and_hardtanh_cuda(
            torch::Tensor x,
            int groups,
            float eps,
            float min_val,
            float max_val
        );
        """

        # Compile with correct header declaration
        self.group_norm_and_hardtanh = load_inline(
            name="group_norm_and_hardtanh",
            cpp_sources=cpp_sources,
            cuda_sources=group_norm_and_hardtanh_source,
            functions=["group_norm_and_hardtanh_cuda"],
            verbose=True,
        )

    def forward(self, x):
        x = self.gemm(x)
        # Replace separate layers with fused kernel
        x = self.group_norm_and_hardtanh.group_norm_and_hardtanh_cuda(
            x,
            self.group_norm.num_groups,
            self.group_norm.eps,
            self.hardtanh.min_val,
            self.hardtanh.max_val,
        )
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]