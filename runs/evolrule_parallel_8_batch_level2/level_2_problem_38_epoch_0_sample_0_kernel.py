#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void fused_clamp_softmax_scale_kernel(
    const float* input,
    float* output,
    const float clamp_min,
    const float clamp_max,
    const float* scale,
    int batch_size,
    int out_channels,
    int spatial_size
) {
    int b = blockIdx.x;
    int c = blockIdx.y;

    int offset = b * out_channels * spatial_size + c * spatial_size;

    __shared__ float s_partials[BLOCK_SIZE];  // Shared memory for reductions

    // Compute local max
    float local_max = -FLT_MAX;
    for (int idx = threadIdx.x; idx < spatial_size; idx += BLOCK_SIZE) {
        float val = input[offset + idx];
        val = min(max(val, clamp_min), clamp_max);
        if (val > local_max) {
            local_max = val;
        }
    }

    // Reduction for max
    s_partials[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_partials[threadIdx.x] < s_partials[threadIdx.x + s]) {
                s_partials[threadIdx.x] = s_partials[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    float global_max = (threadIdx.x == 0) ? s_partials[0] : 0.0f;
    __syncthreads();

    // Compute local sum of exp(val - global_max)
    float local_sum = 0.0f;
    for (int idx = threadIdx.x; idx < spatial_size; idx += BLOCK_SIZE) {
        float val = input[offset + idx];
        val = min(max(val, clamp_min), clamp_max);
        float exp_val = expf(val - global_max);
        local_sum += exp_val;
    }

    // Reduction for sum
    s_partials[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_partials[threadIdx.x] += s_partials[threadIdx.x + s];
        }
        __syncthreads();
    }

    float global_sum = (threadIdx.x == 0) ? s_partials[0] : 0.0f;
    __syncthreads();

    // Compute output
    for (int idx = threadIdx.x; idx < spatial_size; idx += BLOCK_SIZE) {
        float val = input[offset + idx];
        val = min(max(val, clamp_min), clamp_max);
        float exp_val = expf(val - global_max);
        float result = (exp_val / global_sum) * scale[c];
        output[offset + idx] = result;
    }
}

// Host function to call the kernel
torch::Tensor fused_clamp_softmax_scale_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    float clamp_min,
    float clamp_max
) {
    const int batch_size = input.size(0);
    const int out_channels = input.size(1);
    const int d = input.size(2);
    const int h = input.size(3);
    const int w = input.size(4);
    const int spatial_size = d * h * w;

    auto output = torch::empty_like(input);

    const dim3 blocks(batch_size, out_channels);  // Each block handles (b, c)
    const dim3 threads(BLOCK_SIZE);

    // Launch the kernel
    fused_clamp_softmax_scale_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        clamp_min,
        clamp_max,
        scale.data_ptr<float>(),
        batch_size,
        out_channels,
        spatial_size
    );

    cudaDeviceSynchronize();
    return output;
}