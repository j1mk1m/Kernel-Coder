import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple, _pair
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), 
                 padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1, bias=False):
        super(ModelNew, self).__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        output_padding = _triple(output_padding)
        
        # Validate kernel dimensions
        if kernel_size[1] != kernel_size[2]:
            raise ValueError("Kernel width and height must be equal")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Check groups validity
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
            
        # Initialize weights and bias similar to PyTorch's ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Compile the CUDA kernel
        self.conv_transpose3d_cuda = load_inline(
            name="conv_transpose3d",
            cuda_sources=self.get_cuda_source(),
            functions=["conv_transpose3d_forward"],
            verbose=True
        )
        
    def forward(self, x):
        # Compute output shape
        batch_size = x.size(0)
        in_depth, in_height, in_width = x.shape[2], x.shape[3], x.shape[4]
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding
        
        out_depth = (in_depth - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d
        out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h
        out_width = (in_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w
        
        # Launch CUDA kernel
        output = self.conv_transpose3d_cuda.conv_transpose3d_forward(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.output_padding, self.groups,
            out_depth, out_height, out_width
        )
        
        return output
    
    def get_cuda_source(self):
        return f"""
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        
        #define THREADS_PER_BLOCK 256

        __device__ float get_weight(float* weight, int w_in_c, int w_out_c, int w_d, int w_h, int w_w) {{
            return weight[w_in_c * (out_channels/group) * kernel_d * kernel_h * kernel_w + 
                        w_out_c * kernel_d * kernel_h * kernel_w + 
                        w_d * kernel_h * kernel_w + 
                        w_h * kernel_w + 
                        w_w];
        }}

        __global__ void conv_transpose3d_forward_kernel(
            const float* __restrict__ input,
            const float* __restrict__ weight,
            const float* __restrict__ bias,
            float* __restrict__ output,
            int batch_size, int in_channels, int out_channels,
            int in_depth, int in_height, int in_width,
            int kernel_d, int kernel_h, int kernel_w,
            int stride_d, int stride_h, int stride_w,
            int padding_d, int padding_h, int padding_w,
            int output_padding_d, int output_padding_h, int output_padding_w,
            int groups,
            int out_depth, int out_height, int out_width
        ) {{
            // Calculate output coordinates
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * out_channels * out_depth * out_height * out_width)
                return;

            int output_c = idx / (out_depth * out_height * out_width);
            int rest = idx % (out_depth * out_height * out_width);
            int output_d = rest / (out_height * out_width);
            int output_h = (rest % (out_height * out_width)) / out_width;
            int output_w = rest % out_width;

            // Compute input coordinates
            int group = output_c / (out_channels / groups);
            int in_c_base = group * (in_channels / groups);
            int out_c_base = group * (out_channels / groups);

            float val = 0.0;
            for (int in_c = in_c_base; in_c < in_c_base + (in_channels / groups); ++in_c) {{
                for (int kd = 0; kd < kernel_d; ++kd) {{
                    for (int kh = 0; kh < kernel_h; ++kh) {{
                        for (int kw = 0; kw < kernel_w; ++kw) {{
                            // Compute corresponding input position
                            int input_d = (output_d - kd - padding_d) / stride_d;
                            int input_h = (output_h - kh - padding_h) / stride_h;
                            int input_w = (output_w - kw - padding_w) / stride_w;

                            // Check if within input bounds
                            if (input_d < 0 || input_d >= in_depth) continue;
                            if (input_h < 0 || input_h >= in_height) continue;
                            if (input_w < 0 || input_w >= in_width) continue;

                            // Get weight value (transposed)
                            float w = weight[
                                in_c * (out_channels / groups) * kernel_d * kernel_h * kernel_w +
                                (output_c - out_c_base) * kernel_d * kernel_h * kernel_w +
                                kd * kernel_h * kernel_w +
                                kh * kernel_w +
                                kw
                            ];

                            val += input[
                                batch_size * in_channels * (input_d * in_height * in_width) + 
                                in_c * in_depth * in_height * in_width + 
                                input_d * in_height * in_width + 
                                input_h * in_width + 
                                input_w
                            ] * w;
                        }}
                    }}
                }}
            }}

            if (bias != nullptr)
                val += bias[output_c];

            output[idx] = val;
        }}

        torch::Tensor conv_transpose3d_forward(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            std::array<int, 3> stride,
            std::array<int, 3> padding,
            std::array<int, 3> output_padding,
            int groups,
            int out_depth,
            int out_height,
            int out_width
        ) {{
            const int batch_size = input.size(0);
            const int in_channels = input.size(1);
            const int out_channels = weight.size(1) * groups;
            const int in_depth = input.size(2);
            const int in_height = input.size(3);
            const int in_width = input.size(4);
            const int kernel_d = weight.size(2);
            const int kernel_h = weight.size(3);
            const int kernel_w = weight.size(4);

            // Output tensor
            auto output = torch::empty({{batch_size, out_channels, out_depth, out_height, out_width}}, 
                                      input.options());

            dim3 blocks((batch_size * out_channels * out_depth * out_height * out_width + THREADS_PER_BLOCK - 1) 
                        / THREADS_PER_BLOCK);
            dim3 threads(THREADS_PER_BLOCK);

            // Launch kernel
            conv_transpose3d_forward_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                output.data_ptr<float>(),
                batch_size, in_channels, out_channels,
                in_depth, in_height, in_width,
                kernel_d, kernel_h, kernel_w,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                output_padding[0], output_padding[1], output_padding[2],
                groups,
                out_depth, out_height, out_width
            );

            cudaDeviceSynchronize();
            return output;
        }}
        """
    
def get_inputs():
    batch_size = 16
    in_channels = 32
    depth = 64
    width = 64
    height = 64
    x = torch.rand(batch_size, in_channels, depth, width, height).cuda()
    return [x]

def get_init_inputs():
    in_channels = 32
    out_channels = 64
    kernel_size = (3, 5, 5)
    return [in_channels, out_channels, kernel_size]