#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

void conv_transpose3d_kernel(float* input, float* weight, float* bias, float* output, int batch_size, int in_channels, int out_channels, int kernel_size_d, int kernel_size_h, int kernel_size_w, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, int output_padding_d, int output_padding_h, int output_padding_w, int groups) {
    int output_depth = ((input_depth - 1) * stride_d - 2 * padding_d + kernel_size_d + output_padding_d);
    int output_height = ((input_height - 1) * stride_h - 2 * padding_h + kernel_size_h + output_padding_h);
    int output_width = ((input_width - 1) * stride_w - 2 * padding_w + kernel_size_w + output_padding_w);

    for (int n = 0; n < batch_size; ++n) {
        for (int g = 0; g < groups; ++g) {
            for (int oc = 0; oc < out_channels / groups; ++oc) {
                for (int id = 0; id < input_depth; ++id) {
                    for (int ih = 0; ih < input_height; ++ih) {
                        for (int iw = 0; iw < input_width; ++iw) {
                            for (int kd = 0; kd < kernel_size_d; ++kd) {
                                for (int kh = 0; kh < kernel_size_h; ++kh) {
                                    for (int kw = 0; kw < kernel_size_w; ++kw) {
                                        int od = id * stride_d - padding_d + kd;
                                        int oh = ih * stride_h - padding_h + kh;
                                        int ow = iw * stride_w - padding_w + kw;
                                        if (od >= 0 && od < output_depth && oh >= 0 && oh < output_height && ow >= 0 && ow < output_width) {
                                            output[n * out_channels * output_depth * output_height * output_width + g * out_channels / groups * output_depth * output_height * output_width + oc * output_depth * output_height * output_width + od * output_height * output_width + oh * output_width + ow] += input[n * in_channels * input_depth * input_height * input_width + g * in_channels / groups * input_depth * input_height * input_width + (oc * groups + g) * input_depth * input_height * input_width + id * input_height * input_width + ih * input_width + iw] * weight[oc * kernel_size_d * kernel_size_h * kernel_size_w + kd * kernel_size_h * kernel_size_w + kh * kernel_size_w + kw];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        for (int n = 0; n < batch_size; ++n) {
            for (int oc = 0; oc < out_channels; ++oc) {
                output[n * out_channels * output_depth * output_height * output_width + oc * output_depth * output_height * output_width] += bias[oc];
            }
        }
    }
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, int output_padding_d, int output_padding_h, int output_padding_w, int groups) {
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({input.size(0), weight.size(0), input.size(1), input.size(2), input.size(3)}, options);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int kernel_size_d = weight.size(2);
    int kernel_size_h = weight.size(3);
    int kernel_size_w = weight.size(4);

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * input.size(1) * input.size(2) * input.size(3) + block_size - 1) / block_size;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid_dim(num_blocks, 1, 1);
    dim3 block_dim(block_size, 1, 1);

    conv_transpose3d_kernel<<<grid_dim, block_dim, 0, stream>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, kernel_size_d, kernel_size_h, kernel_size_w, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, output_padding_d, output_padding_h, output_padding_w, groups);

    return output;
}