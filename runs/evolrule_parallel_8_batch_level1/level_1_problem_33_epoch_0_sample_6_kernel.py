import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features).cuda())
        self.beta = nn.Parameter(torch.zeros(num_features).cuda())
        self.eps = 1e-5  # same as PyTorch's default epsilon

        compute_mean_var_and_normalize_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void compute_mean_var(
            const scalar_t* input,
            scalar_t* mean,
            scalar_t* var,
            int N, int C, int H, int W) {
            const int c = blockIdx.x;
            if (c >= C) return;

            const int count = N * H * W;
            const int tid = threadIdx.x;

            __shared__ scalar_t s_sum[2][256];
            for (int i = 0; i < 2; ++i) {
                s_sum[i][tid] = 0.0;
            }
            __syncthreads();

            for (int i = tid; i < count; i += blockDim.x) {
                int n = i / (H * W);
                int rem = i % (H * W);
                int h = rem / W;
                int w = rem % W;
                int idx = c * N * H * W + n * H * W + h * W + w;
                scalar_t val = input[idx];
                s_sum[0][tid] += val;
                s_sum[1][tid] += val * val;
            }

            __syncthreads();

            for (int s = blockDim.x/2; s > 0; s >>=1) {
                if (tid < s) {
                    s_sum[0][tid] += s_sum[0][tid + s];
                    s_sum[1][tid] += s_sum[1][tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                mean[c] = s_sum[0][0] / count;
                scalar_t mean_sq = mean[c] * mean[c];
                var[c] = (s_sum[1][0] / count) - mean_sq;
            }
        }

        template <typename scalar_t>
        __global__ void normalize(
            const scalar_t* input,
            scalar_t* output,
            const scalar_t* mean,
            const scalar_t* var,
            scalar_t eps,
            const scalar_t* gamma,
            const scalar_t* beta,
            int N, int C, int H, int W) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N*C*H*W) return;

            int c = (idx / (H * W)) % C;
            int n = idx / (C * H * W);
            int rem = idx % (C * H * W);
            int h = (rem % (H * W)) / W;
            int w = rem % W;

            scalar_t x = input[idx];
            scalar_t inv_std = 1.0f / sqrt(var[c] + eps);
            scalar_t normalized = (x - mean[c]) * inv_std;
            output[idx] = gamma[c] * normalized + beta[c];
        }

        std::tuple<torch::Tensor, torch::Tensor> compute_mean_var_cuda(
            torch::Tensor input,
            int N, int C, int H, int W) {
            const int threads = 256;
            const dim3 blocks(C);

            torch::Tensor mean = torch::zeros({C}, torch::device("cuda").dtype(input.scalar_type()));
            torch::Tensor var = torch::zeros({C}, torch::device("cuda").dtype(input.scalar_type()));

            compute_mean_var<<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                mean.data_ptr<scalar_t>(),
                var.data_ptr<scalar_t>(),
                N, C, H, W
            );

            cudaDeviceSynchronize();
            return std::make_tuple(mean, var);
        }

        std::tuple<torch::Tensor> normalize_cuda(
            torch::Tensor input,
            torch::Tensor mean,
            torch::Tensor var,
            float eps,
            torch::Tensor gamma,
            torch::Tensor beta,
            int N, int C, int H, int W) {
            const int num_elements = N*C*H*W;
            const int threads = 256;
            const int blocks = (num_elements + threads -1) / threads;

            torch::Tensor output = torch::empty_like(input);

            normalize<<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                mean.data_ptr<scalar_t>(),
                var.data_ptr<scalar_t>(),
                eps,
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                N, C, H, W
            );

            cudaDeviceSynchronize();
            return std::make_tuple(output);
        }

        template __global__ void compute_mean_var<float><<<>>>(float*, float*, float*, int, int, int, int);
        template __global__ void normalize<float><<<>>>(float*, float*, float*, float*, float, float*, float*, int, int, int, int);
        """

        self.module = load_inline(
            name="bn_custom",
            cpp_sources="",
            cuda_sources=compute_mean_var_and_normalize_source,
            functions=["compute_mean_var_cuda", "normalize_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        mean, var = self.module.compute_mean_var_cuda(x, N, C, H, W)
        output = self.module.normalize_cuda(
            x,
            mean,
            var,
            self.eps,
            self.gamma,
            self.beta,
            N, C, H, W
        )[0]
        return output