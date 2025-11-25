import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        # Initialize the transposed convolution layer
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, bias=bias
        )
        
        # Define the fused softmax-sigmoid kernel
        fused_softmax_sigmoid_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <mma.h>

        __global__ void fused_softmax_sigmoid_kernel(
            const float* input,
            float* output,
            int batch_size,
            int channels,
            int depth,
            int height,
            int width
        ) {
            int batch = blockIdx.x;
            int d = blockIdx.y;
            int h = blockIdx.z;
            int w = threadIdx.x;

            // Compute the input and output offsets
            int input_offset = batch * channels * depth * height * width +
                              d * height * width +
                              h * width +
                              w;
            int output_offset = batch * channels * depth * height * width +
                               d * height * width +
                               h * width +
                               w;

            // Compute softmax over channels
            float sum_exp = 0.0;
            for (int c = 0; c < channels; ++c) {
                int idx = input_offset + c * depth * height * width;
                sum_exp += exp(input[idx]);
            }

            // Compute softmax and sigmoid in parallel
            for (int c = 0; c < channels; ++c) {
                int idx = input_offset + c * depth * height * width;
                float softmax_val = exp(input[idx]) / sum_exp;
                float sigmoid_val = 1.0 / (1.0 + exp(-input[idx]));
                output[output_offset + c * depth * height * width] = softmax_val * sigmoid_val;
            }
        }

        torch::Tensor fused_softmax_sigmoid_cuda(torch::Tensor input) {
            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int depth = input.size(2);
            const int height = input.size(3);
            const int width = input.size(4);

            auto output = torch::empty_like(input);

            dim3 grid(batch_size, depth, height);
            dim3 block(width);

            fused_softmax_sigmoid_kernel<<<grid, block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                channels,
                depth,
                height,
                width
            );

            return output;
        }
        """

        # Compile the fused kernel
        fused_softmax_sigmoid = load_inline(
            name="fused_softmax_sigmoid",
            cuda_sources=fused_softmax_sigmoid_source,
            functions=["fused_softmax_sigmoid_cuda"],
            verbose=False
        )

        # Assign the kernel to the module
        self.fused_softmax_sigmoid = fused_softmax_sigmoid

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_softmax_sigmoid.fused_softmax_sigmoid_cuda(x)
        return x

# Ensure compatibility with original initialization
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]