import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_source = """
extern "C"{
    __global__ void conv2d_cuda_kernel(
        const float* __restrict__ input,
        const float* __restrict__ kernel,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int batch_size,
        int in_channels,
        int out_channels,
        int input_height,
        int input_width,
        int kernel_h,
        int kernel_w,
        int stride,
        int padding_h,
        int padding_w,
        int dilation_h,
        int dilation_w,
        int output_height,
        int output_width) {

        int b = blockIdx.x;
        int h_out = blockIdx.y;
        int w_out = blockIdx.z;
        int oc = threadIdx.x;

        int h_start = h_out * stride;
        int w_start = w_out * stride;

        int h_region = (kernel_h - 1) * dilation_h + 1;
        int w_region = (kernel_w - 1) * dilation_w + 1;

        extern __shared__ float s_input[];

        int num_elements = in_channels * h_region * w_region;
        int tid = threadIdx.x;

        for (int idx = tid; idx < num_elements; idx += blockDim.x) {
            int ic = idx / (h_region * w_region);
            int row_col_idx = idx % (h_region * w_region);
            int row_in_region = row_col_idx / w_region;
            int col_in_region = row_col_idx % w_region;

            int padded_h = h_start + row_in_region;
            int padded_w = w_start + col_in_region;

            bool valid = (padded_h >= 0 && padded_h < input_height + 2*padding_h) &&
                         (padded_w >= 0 && padded_w < input_width + 2*padding_w);

            int actual_h = padded_h - padding_h;
            int actual_w = padded_w - padding_w;

            bool input_valid = (actual_h >= 0 && actual_h < input_height) &&
                              (actual_w >= 0 && actual_w < input_width);

            float val = 0.0f;
            if (valid && input_valid) {
                val = input[b * in_channels * input_height * input_width +
                           ic * input_height * input_width +
                           actual_h * input_width + actual_w];
            }

            s_input[idx] = val;
        }

        __syncthreads();

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int row_in_region = kh * dilation_h;
                    int col_in_region = kw * dilation_w;

                    int s_idx = ic * h_region * w_region +
                                row_in_region * w_region + col_in_region;

                    float kernel_val = kernel[oc * in_channels * kernel_h * kernel_w +
                                             ic * kernel_h * kernel_w +
                                             kh * kernel_w + kw];

                    sum += kernel_val * s_input[s_idx];
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[oc];
        }

        output[b * out_channels * output_height * output_width +
               oc * output_height * output_width +
               h_out * output_width + w_out] = sum;
    }
}
"""

conv2d_cpp_source = "void conv2d_cuda_kernel(...);"

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        # Initialize weights
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Load CUDA kernel
        self.conv2d_cuda_kernel = load_inline(
            name="conv2d_cuda",
            cuda_sources=conv2d_source,
            functions=["conv2d_cuda_kernel"],
            verbose=True
        )

    def forward(self, x):
        batch_size, in_channels, input_height, input_width = x.size()
        kernel_h, kernel_w = self.kernel.size()[2:]
        out_channels = self.kernel.size(0)
        stride = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation

        H_padded = input_height + 2 * padding_h
        W_padded = input_width + 2 * padding_w
        effective_kernel_h = (kernel_h - 1) * dilation_h + 1
        effective_kernel_w = (kernel_w - 1) * dilation_w + 1
        output_height = (H_padded - effective_kernel_h) // stride + 1
        output_width = (W_padded - effective_kernel_w) // stride + 1

        output = torch.empty(batch_size, out_channels, output_height, output_width, device=x.device)

        h_region = (kernel_h - 1) * dilation_h + 1
        w_region = (kernel_w - 1) * dilation_w + 1
        shared_mem_size = in_channels * h_region * w_region * 4  # bytes

        block_size = out_channels
        block = (block_size, 1, 1)
        grid = (batch_size, output_height, output_width)

        kernel_ptr = self.kernel.contiguous().data_ptr()
        bias_ptr = self.bias.contiguous().data_ptr() if self.bias is not None else 0

        kernel_args = [
            x.contiguous().data_ptr(),
            kernel_ptr,
            bias_ptr,
            output.data_ptr(),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            kernel_h,
            kernel_w,
            stride,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output_height,
            output_width
        ]

        self.conv2d_cuda_kernel.conv2d_cuda_kernel(
            grid=grid,
            block=block,
            shared_mem=shared_mem_size,
            stream=torch.cuda.current_stream().cuda_stream,
            args=kernel_args
        )

        return output