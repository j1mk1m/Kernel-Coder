import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.num_features = num_features

        instance_norm_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void compute_mean_var(
            const float* x,
            float* sums,
            float* sum_sqs,
            int N, int C, int H, int W
        ) {
            int n = blockIdx.y;
            int c = blockIdx.x;
            int tid = threadIdx.x;

            float partial_sum = 0.0f;
            float partial_sum_sq = 0.0f;

            int total_elements = H * W;
            for (int i = tid; i < total_elements; i += blockDim.x) {
                int h = i / W;
                int w = i % W;
                int offset = n * C * H * W + c * H * W + h * W + w;
                float val = x[offset];
                partial_sum += val;
                partial_sum_sq += val * val;
            }

            extern __shared__ float s[];
            float* s_partial_sum = s;
            float* s_partial_sum_sq = s + blockDim.x;

            s_partial_sum[tid] = partial_sum;
            s_partial_sum_sq[tid] = partial_sum_sq;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    s_partial_sum[tid] += s_partial_sum[tid + s];
                    s_partial_sum_sq[tid] += s_partial_sum_sq[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                sums[n * C + c] = s_partial_sum[0];
                sum_sqs[n * C + c] = s_partial_sum_sq[0];
            }
        }

        __global__ void apply_normalization(
            const float* x,
            float* y,
            const float* sums,
            const float* sum_sqs,
            const float* gamma,
            const float* beta,
            int N, int C, int H, int W
        ) {
            int n = blockIdx.y;
            int c = blockIdx.x;
            int tid = threadIdx.x;

            int size = H * W;
            float sum = sums[n * C + c];
            float sum_sq = sum_sqs[n * C + c];
            float mean = sum / (float)size;
            float var = (sum_sq / (float)size) - mean * mean;
            var += 1e-5f;
            float denom = rsqrt(var);

            for (int i = tid; i < size; i += blockDim.x) {
                int h = i / W;
                int w = i % W;
                int offset = n * C * H * W + c * H * W + h * W + w;
                float val = x[offset];
                float normalized = (val - mean) * denom;
                y[offset] = normalized * gamma[c] + beta[c];
            }
        }

        torch::Tensor instance_norm_cuda(
            torch::Tensor x,
            torch::Tensor gamma,
            torch::Tensor beta
        ) {
            const int N = x.size(0);
            const int C = x.size(1);
            const int H = x.size(2);
            const int W = x.size(3);

            auto y = torch::empty_like(x);
            auto sums = torch::empty({N * C}, x.options());
            auto sum_sqs = torch::empty({N * C}, x.options());

            const int block_size = 256;
            const dim3 grid_dim(C, N);
            const dim3 block_dim(block_size);

            AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "instance_norm_cuda", ([&] {
                compute_mean_var<<<grid_dim, block_dim, 2 * block_size * sizeof(float)>>>(
                    x.data_ptr<scalar_t>(),
                    sums.data_ptr<float>(),
                    sum_sqs.data_ptr<float>(),
                    N, C, H, W
                );

                apply_normalization<<<grid_dim, block_dim>>>(
                    x.data_ptr<scalar_t>(),
                    y.data_ptr<scalar_t>(),
                    sums.data_ptr<float>(),
                    sum_sqs.data_ptr<float>(),
                    gamma.data_ptr<float>(),
                    beta.data_ptr<float>(),
                    N, C, H, W
                );
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
            }

            return y;
        }
        """

        instance_norm_cpp_source = """
        torch::Tensor instance_norm_cuda(
            torch::Tensor x,
            torch::Tensor gamma,
            torch::Tensor beta
        );
        """

        self.instance_norm_cuda = load_inline(
            name="instance_norm_cuda",
            cpp_sources=instance_norm_cpp_source,
            cuda_sources=instance_norm_source,
            functions=["instance_norm_cuda"],
            verbose=True,
            extra_cflags=["-DVERSION_GE_1_5"],
            extra_cuda_cflags=["--expt-extended-lambda"],
        )

    def forward(self, x):
        device = self.gamma.device
        x = x.to(device)
        gamma = self.gamma.to(device)
        beta = self.beta.to(device)
        return self.instance_norm_cuda.instance_norm_cuda(x, gamma, beta)