import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        compute_sums_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void compute_group_sums(
            const float* input,
            float* sum_gs,
            float* sum_squares_gs,
            int G,
            int C,
            int N,
            int spatial_size,
            int C_per_group,
            int elements_per_gs
        ) {
            int g = blockIdx.x;
            int n = blockIdx.y;
            if (g >= G || n >= N) return;

            extern __shared__ float shared[];
            float* sdata = shared;
            int tid = threadIdx.x;

            float partial_sum = 0.0f;
            float partial_squares = 0.0f;

            int offset = n * C * spatial_size + g * C_per_group * spatial_size;
            for (int i = tid; i < elements_per_gs; i += blockDim.x) {
                int idx = offset + i;
                float val = input[idx];
                partial_sum += val;
                partial_squares += val * val;
            }

            sdata[tid] = partial_sum;
            sdata[tid + blockDim.x] = partial_squares;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                    sdata[tid + blockDim.x] += sdata[tid + s + blockDim.x];
                }
                __syncthreads();
            }

            if (tid == 0) {
                sum_gs[g * N + n] = sdata[0];
                sum_squares_gs[g * N + n] = sdata[blockDim.x];
            }
        }

        __global__ void apply_group_norm(
            const float* input,
            float* output,
            const float* means,
            const float* vars,
            const float* gamma,
            const float* beta,
            int G,
            int C,
            int N,
            int spatial_size,
            int C_per_group,
            float eps
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N * C * spatial_size) return;

            int n = idx / (C * spatial_size);
            int rem = idx % (C * spatial_size);
            int c = rem / spatial_size;
            int spatial = rem % spatial_size;

            int g = c / C_per_group;
            int mn_index = g * N + n;
            float mean = means[mn_index];
            float var = vars[mn_index];
            float g_val = gamma[c];
            float b_val = beta[c];

            float x = input[idx];
            float normed = (x - mean) / sqrtf(var + eps);
            output[idx] = normed * g_val + b_val;
        }
        """

        module = load_inline(
            name="group_norm_cuda",
            cpp_sources="",
            cuda_sources=compute_sums_source,
            functions=[
                "void compute_group_sums(const float*, float*, float*, int, int, int, int, int, int)",
                "void apply_group_norm(const float*, float*, const float*, const float*, const float*, const float*, int, int, int, int, int, float)"
            ],
            verbose=True
        )

        self.compute_group_sums = module.compute_group_sums
        self.apply_group_norm = module.apply_group_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C = x.size(0), x.size(1)
        spatial_dims = x.size()[2:]
        spatial_size = 1
        for s in spatial_dims:
            spatial_size *= s

        G = self.num_groups
        C_per_group = C // G
        elements_per_gs = C_per_group * spatial_size
        total_elements = N * C * spatial_size

        sum_gs = torch.zeros(G * N, dtype=x.dtype, device=x.device)
        sum_squares_gs = torch.zeros_like(sum_gs)

        block_size = 256
        block = (block_size, 1)
        grid = (G, N)
        shared_size = block_size * 2 * torch.cuda.FloatTensor().element_size()

        self.compute_group_sums(
            grid=grid,
            block=block,
            shared_mem_per_block=shared_size,
            stream=torch.cuda.current_stream().cuda_stream,
            args=[
                x.data_ptr(),
                sum_gs.data_ptr(),
                sum_squares_gs.data_ptr(),
                G, C, N, spatial_size, C_per_group, elements_per_gs
            ]
        )

        means = torch.empty_like(sum_gs)
        vars_ = torch.empty_like(sum_gs)
        for g in range(G):
            for n in range(N):
                idx = g * N + n
                s = sum_gs[idx].item()
                s_sq = sum_squares_gs[idx].item()
                mean_val = s / elements_per_gs
                var_val = (s_sq / elements_per_gs) - (mean_val * mean_val)
                means[idx] = mean_val
                vars_[idx] = var_val

        output = torch.empty_like(x)
        block_norm = (256, 1)
        grid_norm = ( (total_elements + block_norm[0] - 1) // block_norm[0], 1 )
        self.apply_group_norm(
            grid=grid_norm,
            block=block_norm,
            stream=torch.cuda.current_stream().cuda_stream,
            args=[
                x.data_ptr(),
                output.data_ptr(),
                means.data_ptr(),
                vars_.data_ptr(),
                self.gamma.data_ptr(),
                self.beta.data_ptr(),
                G, C, N, spatial_size, C_per_group, 1e-5
            ]
        )
        return output

# Original code's functions remain unchanged
batch_size = 112
features = 64
num_groups = 8
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features, num_groups]