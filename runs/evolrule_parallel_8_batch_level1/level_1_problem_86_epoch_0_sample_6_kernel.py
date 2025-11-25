#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_X 16
#define BLOCK_Y 16

template <bool HAS_BIAS>
__global__ void fused_convolution(
    const float* __restrict__ input,
    const float* __restrict__ depthwise_weights,
    const float* __restrict__ pointwise_weights,
    const float* __restrict__ depthwise_biases,
    const float* __restrict__ pointwise_biases,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_size, int stride, int padding, int dilation,
    int output_height, int output_width) {

    int n = blockIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_y >= output_height || out_x >= output_width) return;

    int in_y = out_y * stride - padding;
    int in_x = out_x * stride - padding;

    float depthwise_results[in_channels];
    for (int c = 0; c < in_channels; c++) {
        depthwise_results[c] = 0.0f;
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int dy = in_y + ky * dilation;
                int dx = in_x + kx * dilation;
                if (dy >= 0 && dy < input_height && dx >= 0 && dx < input_width) {
                    int input_offset = n * in_channels * input_height * input_width
                                      + c * input_height * input_width
                                      + dy * input_width + dx;
                    int weight_offset = c * kernel_size * kernel_size
                                       + ky * kernel_size + kx;
                    depthwise_results[c] += input[input_offset] * depthwise_weights[weight_offset];
                }
            }
        }
        if (HAS_BIAS) {
            depthwise_results[c] += depthwise_biases[c];
        }
    }

    for (int p_out = 0; p_out < out_channels; p_out++) {
        float sum = 0.0f;
        for (int p_in = 0; p_in < in_channels; p_in++) {
            int weight_offset = p_out * in_channels + p_in;
            sum += depthwise_results[p_in] * pointwise_weights[weight_offset];
        }
        if (HAS_BIAS) {
            sum += pointwise_biases[p_out];
        }

        int output_offset = n * out_channels * output_height * output_width
                           + p_out * output_height * output_width
                           + out_y * output_width + out_x;
        output[output_offset] = sum;
    }
}

torch::Tensor fused_convolution_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weights,
    torch::Tensor pointwise_weights,
    torch::Tensor depthwise_biases,
    torch::Tensor pointwise_biases,
    int stride, int padding, int dilation) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int kernel_size = depthwise_weights.size(2);
    int out_channels = pointwise_weights.size(0);

    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto options = input.options();
    torch::Tensor output = torch::zeros({batch_size, out_channels, output_height, output_width}, options);

    dim3 threads(BLOCK_X, BLOCK_Y);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size
    );

    bool has_bias = !depthwise_biases.defined() ? false : true;

    if (has_bias) {
        fused_convolution<true><<<blocks, threads>>>(
            input.data_ptr<float>(),
            depthwise_weights.data_ptr<float>(),
            pointwise_weights.data_ptr<float>(),
            depthwise_biases.data_ptr<float>(),
            pointwise_biases.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_size, stride, padding, dilation,
            output_height, output_width);
    } else {
        fused_convolution<false><<<blocks, threads>>>(
            input.data_ptr<float>(),
            depthwise_weights.data_ptr<float>(),
            pointwise_weights.data_ptr<float>(),
            nullptr, nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_size, stride, padding, dilation,
            output_height, output_width);
    }

    cudaDeviceSynchronize();
    return output;
}