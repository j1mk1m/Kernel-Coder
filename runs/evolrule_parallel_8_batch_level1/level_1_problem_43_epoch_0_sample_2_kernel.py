import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

        # Load the CUDA kernel
        max_pool3d_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void max_pool3d_forward(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            int64_t* __restrict__ indices,
            int batch, int channels,
            int idim1, int idim2, int idim3,
            int odim1, int odim2, int odim3,
            int kernel_size, int stride,
            int padding, int dilation,
            bool return_indices, bool ceil_mode) {

            int batch_idx = blockIdx.x;
            int channel_idx = blockIdx.y;
            int out_d = threadIdx.z;
            int out_h = threadIdx.y;
            int out_w = threadIdx.x;

            // Calculate output position
            int out_idx = out_d * odim2 * odim3 + out_h * odim3 + out_w;
            int input_offset = batch_idx * channels * idim1 * idim2 * idim3 + channel_idx * idim1 * idim2 * idim3;

            // Compute input indices with padding and dilation
            int in_d_start = -padding;
            int in_h_start = -padding;
            int in_w_start = -padding;
            for (int k = 0; k < kernel_size; ++k) {
                in_d_start += dilation * k;
                in_h_start += dilation * k;
                in_w_start += dilation * k;
            }

            // Output dimensions calculation based on ceil_mode
            int in_d = out_d * stride + in_d_start;
            int in_h = out_h * stride + in_h_start;
            int in_w = out_w * stride + in_w_start;

            // Check if within bounds
            bool valid = (in_d >= 0 && in_d < idim1 &&
                        in_h >= 0 && in_h < idim2 &&
                        in_w >= 0 && in_w < idim3);

            scalar_t max_val = -FLT_MAX;
            int max_idx = -1;
            int current_idx = 0;

            for (int d = 0; d < kernel_size; ++d) {
                for (int h = 0; h < kernel_size; ++h) {
                    for (int w = 0; w < kernel_size; ++w) {
                        int dd = in_d + dilation * d;
                        int hh = in_h + dilation * h;
                        int ww = in_w + dilation * w;
                        if (dd < 0 || dd >= idim1 || hh < 0 || hh >= idim2 || ww < 0 || ww >= idim3) {
                            continue;
                        }
                        scalar_t val = input[input_offset + dd * idim2 * idim3 + hh * idim3 + ww];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = current_idx;
                        }
                        current_idx++;
                    }
                }
            }

            if (return_indices) {
                indices[batch_idx * channels * odim1 * odim2 * odim3 + channel_idx * odim1 * odim2 * odim3 + out_idx] = max_idx;
            }
            output[batch_idx * channels * odim1 * odim2 * odim3 + channel_idx * odim1 * odim2 * odim3 + out_idx] = max_val;
        }

        std::tuple<torch::Tensor, torch::Tensor> max_pool3d_forward_cuda(torch::Tensor input,
            int kernel_size, int stride, int padding, int dilation, bool return_indices, bool ceil_mode) {

            const auto batch = input.size(0);
            const auto channels = input.size(1);
            const auto idim1 = input.size(2);
            const auto idim2 = input.size(3);
            const auto idim3 = input.size(4);

            // Compute output dimensions
            auto compute_output_dim = [](int input_dim, int kernel, int stride, int padding, bool ceil) {
                int output_dim = (input_dim + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
                if (ceil) {
                    output_dim = (input_dim + 2 * padding - 1) / stride - dilation * (kernel - 1);
                }
                return output_dim;
            };

            int odim1 = compute_output_dim(idim1, kernel_size, stride, padding, ceil_mode);
            int odim2 = compute_output_dim(idim2, kernel_size, stride, padding, ceil_mode);
            int odim3 = compute_output_dim(idim3, kernel_size, stride, padding, ceil_mode);

            auto output = torch::empty({batch, channels, odim1, odim2, odim3}, input.options());
            auto indices = torch::empty({batch, channels, odim1, odim2, odim3}, torch::kInt64);

            dim3 threads(kernel_size, kernel_size, kernel_size);
            dim3 blocks(batch, channels, 1);

            AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool3d_forward", ([&] {
                max_pool3d_forward<scalar_t><<<blocks, threads>>>(
                    input.data<scalar_t>(),
                    output.data<scalar_t>(),
                    indices.data<int64_t>(),
                    batch, channels, idim1, idim2, idim3,
                    odim1, odim2, odim3,
                    kernel_size, stride, padding, dilation,
                    return_indices, ceil_mode);
            }));

            if (!return_indices) {
                return std::tuple<torch::Tensor, torch::Tensor>(output, torch::Tensor());
            }
            return std::make_tuple(output, indices);
        }
        """
        max_pool3d_cpp_source = """
        std::tuple<torch::Tensor, torch::Tensor> max_pool3d_forward_cuda(torch::Tensor input,
            int kernel_size, int stride, int padding, int dilation, bool return_indices, bool ceil_mode);
        """

        self.forward_op = load_inline(
            name="max_pool3d_forward",
            cpp_sources=max_pool3d_cpp_source,
            cuda_sources=max_pool3d_source,
            functions=["max_pool3d_forward_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_op.max_pool3d_forward_cuda(
            x.cuda(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
            self.ceil_mode
        )
        return output

# Ensure the get_inputs and get_init_inputs are provided as per original
batch_size = 16
channels = 32
dim1 = 128
dim2 = 128
dim3 = 128
kernel_size = 3
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.rand(batch_size, channels, dim1, dim2, dim3).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]