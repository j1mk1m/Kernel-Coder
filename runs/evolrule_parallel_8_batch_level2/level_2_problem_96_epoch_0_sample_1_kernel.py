#include <torch/extension.h>
#include <cuda_runtime.h>

template <int BlockSize>
__global__ void global_avg_pool3d_clamp_kernel(
    const float* input, float* output,
    int batch_size, int channels,
    int depth, int height, int width,
    float min_val, float max_val) {

    int b = blockIdx.x;
    int c = blockIdx.y;

    if (b >= batch_size || c >= channels) return;

    int base = b * channels * depth * height * width +
               c * depth * height * width;

    int tid = threadIdx.x;
    int total_spatial = depth * height * width;

    extern __shared__ float shared[];

    float sum = 0.0f;

    // Each thread processes a chunk of the spatial indices
    for (int spatial_idx = tid; spatial_idx < total_spatial; spatial_idx += BlockSize) {
        sum += input[base + spatial_idx];
    }

    __syncthreads();

    // Store partial sum in shared memory
    if (tid < BlockSize) {
        shared[tid] = sum;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = BlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float avg = shared[0] / total_spatial;
        avg = fmaxf(fminf(avg, max_val), min_val);
        int output_offset = b * channels + c;
        output[output_offset] = avg;
    }
}

torch::Tensor global_avg_pool3d_clamp_cuda(
    torch::Tensor input,
    float min_val,
    float max_val) {

    TORCH_CHECK(input.dim() == 5, "Input must be 5D.");

    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    auto output = torch::empty({batch_size, channels, 1, 1, 1}, input.options());

    const int block_size = 256;
    dim3 blocks(batch_size, channels);
    int shared_size = block_size * sizeof(float);

    global_avg_pool3d_clamp_kernel<block_size><<<blocks, block_size, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        min_val,
        max_val
    );

    return output;
}