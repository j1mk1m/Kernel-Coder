import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, bias=bias
        )
        
        # Define fused softmax-sigmoid CUDA kernel
        fused_kernel_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void fused_softmax_sigmoid_kernel(
            const float* input,
            float* output,
            int batch_size,
            int channels,
            int D,
            int H,
            int W) {
            int batch = blockIdx.x / (D * H * W);
            int spatial_pos = blockIdx.x % (D * H * W);
            int d = spatial_pos / (H * W);
            int hw = spatial_pos % (H * W);
            int h = hw / W;
            int w = hw % W;

            int channel = threadIdx.x;

            if (channel >= channels) return;

            int input_offset = batch * channels * D * H * W +
                              channel * D * H * W +
                              d * H * W + h * W + w;

            float z = input[input_offset];
            float exp_z = expf(z);

            __shared__ float shared_data[64]; // channels=64 as per problem parameters
            shared_data[threadIdx.x] = exp_z;
            __syncthreads();

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
                }
                __syncthreads();
            }

            float sum_exp = shared_data[0];
            float softmax_val = exp_z / sum_exp;
            float sigmoid_val = 1.0f / (1.0f + expf(-softmax_val));

            output[input_offset] = sigmoid_val;
        }

        torch::Tensor fused_softmax_sigmoid(torch::Tensor input) {
            auto batch_size = input.size(0);
            auto channels = input.size(1);
            auto D = input.size(2);
            auto H = input.size(3);
            auto W = input.size(4);

            auto output = torch::empty_like(input);

            dim3 block_size(channels);
            dim3 grid_size(batch_size * D * H * W);

            fused_softmax_sigmoid_kernel<<<grid_size, block_size>>>(
                input.contiguous().data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size, channels, D, H, W
            );

            return output;
        }
        """
        
        cpp_header = """
        torch::Tensor fused_softmax_sigmoid(torch::Tensor input);
        """
        
        # Compile the fused kernel
        self.fused_mod = load_inline(
            name="fused_softmax_sigmoid",
            cpp_sources=cpp_header,
            cuda_sources=fused_kernel_source,
            functions=["fused_softmax_sigmoid"],
            verbose=True
        )
        self.fused_activation = self.fused_mod.fused_softmax_sigmoid

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_activation(x)
        return x

# The get_inputs and get_init_inputs remain unchanged as per original code
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]