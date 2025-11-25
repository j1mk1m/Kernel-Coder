#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void group_norm_kernel(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* gamma,
    const scalar_t* beta,
    int N, int C, int D, int H, int W,
    int groups, float eps) {

    // blockIdx.x is sample*groups + group
    int sample = blockIdx.x / groups;
    int group = blockIdx.x % groups;

    int channels_per_group = C / groups;
    int start_channel = group * channels_per_group;
    int end_channel = start_channel + channels_per_group;

    int total_elements = channels_per_group * D * H * W;

    extern __shared__ float shared[];
    float* s_sum = (float*)shared;
    float* s_sum_sq = s_sum + blockDim.x;

    s_sum[threadIdx.x] = 0.0f;
    s_sum_sq[threadIdx.x] = 0.0f;
    __syncthreads();

    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        int c_in_group = i / (D * H * W);
        int d = (i / (H * W)) % D;
        int h = (i / W) % H;
        int w = i % W;

        int c = start_channel + c_in_group;
        int idx = sample * C * D * H * W + c * D * H * W + d * H * W + h * W + w;

        scalar_t val = input[idx];
        s_sum[threadIdx.x] += val;
        s_sum_sq[threadIdx.x] += val * val;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean = s_sum[0] / total_elements;
        float var = s_sum_sq[0] / total_elements - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);

        s_sum[0] = mean;
        s_sum_sq[0] = inv_std;
    }
    __syncthreads();

    float mean = s_sum[0];
    float inv_std = s_sum_sq[0];

    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        int c_in_group = i / (D * H * W);
        int d = (i / (H * W)) % D;
        int h = (i / W) % H;
        int w = i % W;

        int c = start_channel + c_in_group;
        int idx = sample * C * D * H * W + c * D * H * W + d * H * W + h * W + w;

        scalar_t val = input[idx];
        scalar_t norm_val = (val - mean) * inv_std;
        norm_val = norm_val * gamma[c] + beta[c];

        output[idx] = norm_val;
    }
}

at::Tensor group_norm_cuda(
    const at::Tensor& input,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    int groups,
    float eps) {

    AT_ASSERT(input.is_contiguous());
    AT_ASSERT(input.dim() == 5);

    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    at::Tensor output = at::empty_like(input);
    dim3 threads(256);
    dim3 blocks(N * groups);  // Each block handles a sample and group

    int shared_size = threads.x * 2 * sizeof(float);
    group_norm_kernel<float><<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N, C, D, H, W,
        groups, eps);

    return output;
}