#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_scale_clamp_kernel(
    const float* input, const float* scale,
    float* output, int num_channels, int spatial_size,
    float clamp_min, float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_size) return; // Wait, input_size is not a parameter here.

    // Need to have input_size as a parameter.

Wait, in the previous example, the kernel used size as an argument. So in this case, the kernel function should have an input_size parameter.

Let me adjust:

__global__ void fused_scale_clamp_kernel(
    const float* input, const float* scale,
    float* output, int input_size, int num_channels, int spatial_size,
    float clamp_min, float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_size) return;

    int channel = (idx / spatial_size) % num_channels;
    float val = input[idx] * scale[channel];
    val = fmaxf(val, clamp_min);
    val = fminf(val, clamp_max);
    output[idx] = val;
}