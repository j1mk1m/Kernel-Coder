import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias similar to PyTorch's Conv3d
        kernel_dims = (out_channels, in_channels // groups) + kernel_size
        self.weight = nn.Parameter(torch.empty(kernel_dims))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Compile custom Conv3D CUDA kernel
        self.custom_conv3d = load_inline(
            name="custom_conv3d",
            cpp_sources=self._get_cpp_source(),
            cuda_sources=self._get_cuda_source(),
            functions=["conv3d_forward"],
            verbose=False
        )

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _get_cpp_source(self):
        return """
        torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                    int stride, int padding, int dilation, int groups);
        """

    def _get_cuda_source(self):
        return """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <vector>

        using torch::Tensor;

        template <typename scalar_t>
        __global__ void conv3d_kernel(const scalar_t* __restrict__ input,
                                     const scalar_t* __restrict__ weight,
                                     scalar_t* __restrict__ output,
                                     const int batch_size,
                                     const int in_channels,
                                     const int out_channels,
                                     const int input_depth,
                                     const int input_height,
                                     const int input_width,
                                     const int kernel_depth,
                                     const int kernel_height,
                                     const int kernel_width,
                                     const int stride,
                                     const int padding,
                                     const int dilation,
                                     const int groups) {
            const int out_depth = (input_depth + 2 * padding - dilation * (kernel_depth - 1) - 1) / stride + 1;
            const int out_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
            const int out_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

            const int output_channels_per_group = out_channels / groups;
            const int in_channels_per_group = in_channels / groups;

            // Calculate output position
            const int w = blockIdx.x * blockDim.x + threadIdx.x;
            const int h = blockIdx.y * blockDim.y + threadIdx.y;
            const int d = blockIdx.z * blockDim.z + threadIdx.z;
            if (w >= out_width || h >= out_height || d >= out_depth) {
                return;
            }

            // Iterate over batch and groups
            for (int n = 0; n < batch_size; ++n) {
                for (int g = 0; g < groups; ++g) {
                    const int out_channel_start = g * output_channels_per_group;
                    for (int oc = 0; oc < output_channels_per_group; ++oc) {
                        const int oc_total = out_channel_start + oc;
                        scalar_t sum = 0;
                        // Iterate over kernel dimensions
                        for (int kd = 0; kd < kernel_depth; ++kd) {
                            const int id = d * stride - padding + kd * dilation;
                            if (id < 0 || id >= input_depth) continue;
                            for (int kh = 0; kh < kernel_height; ++kh) {
                                const int ih = h * stride - padding + kh * dilation;
                                if (ih < 0 || ih >= input_height) continue;
                                for (int kw = 0; kw < kernel_width; ++kw) {
                                    const int iw = w * stride - padding + kw * dilation;
                                    if (iw < 0 || iw >= input_width) continue;
                                    // Compute input and weight indices
                                    const int in_offset = n * in_channels * input_depth * input_height * input_width
                                                        + (g * in_channels_per_group + (kd * kernel_height + kh) * kernel_width + kw)
                                                        + id * input_height * input_width + ih * input_width + iw;
                                    const int weight_offset = oc_total * in_channels_per_group * kernel_depth * kernel_height * kernel_width
                                                            + (kd * kernel_height + kh) * kernel_width + kw;
                                    sum += input[in_offset] * weight[weight_offset];
                                }
                            }
                        }
                        // Store result
                        const int out_offset = n * out_channels * out_depth * out_height * out_width
                                            + oc_total * out_depth * out_height * out_width
                                            + d * out_height * out_width + h * out_width + w;
                        output[out_offset] = sum;
                    }
                }
            }
        }

        torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                    int stride, int padding, int dilation, int groups) {
            const auto batch_size = input.size(0);
            const auto in_channels = input.size(1);
            const auto input_depth = input.size(2);
            const auto input_height = input.size(3);
            const auto input_width = input.size(4);

            const auto kernel_depth = weight.size(2);
            const auto kernel_height = weight.size(3);
            const auto kernel_width = weight.size(4);

            const auto out_depth = (input_depth + 2 * padding - dilation * (kernel_depth - 1) - 1) / stride + 1;
            const auto out_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
            const auto out_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

            auto output = torch::zeros({batch_size, weight.size(0), out_depth, out_height, out_width}, input.options());

            dim3 threads(16, 16, 1);
            dim3 blocks((out_width + threads.x - 1) / threads.x,
                       (out_height + threads.y - 1) / threads.y,
                       (out_depth + threads.z - 1) / threads.z);

            AT_DISPATCH_ALL_TYPES(input.scalar_type(), "conv3d_forward", ([&] {
                conv3d_kernel<scalar_t><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, in_channels, weight.size(0),
                    input_depth, input_height, input_width,
                    kernel_depth, kernel_height, kernel_width,
                    stride, padding, dilation, groups);
            }));

            if (bias.defined()) {
                output += bias.view(1, -1, 1, 1, 1);
            }

            return output;
        }
        """

    def forward(self, x):
        return self.custom_conv3d.conv3d_forward(
            x, self.weight, self.bias if self.bias is not None else x.new_zeros(0),
            self.stride, self.padding, self.dilation, self.groups
        )