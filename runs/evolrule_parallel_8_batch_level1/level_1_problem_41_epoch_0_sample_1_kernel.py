import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.return_indices = return_indices

        # Load the custom CUDA kernel
        max_pool1d_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <vector>

        template <typename scalar_t>
        __global__ void max_pool1d_kernel(
            const torch::PackedTensorAccessor<scalar_t,3> input,
            torch::PackedTensorAccessor<scalar_t,3> output,
            torch::PackedTensorAccessor<int64_t,3> indices,
            const int batch_size,
            const int channels,
            const int in_length,
            const int padding_l,
            const int kernel_size,
            const int stride,
            const int dilation) {

            const int batch_idx = blockIdx.x;
            const int channel_idx = blockIdx.y;
            const int out_pos = threadIdx.x;

            const int output_length = output.size(2);
            if (out_pos >= output_length) return;

            scalar_t max_val = -FLT_MAX;
            int max_idx = -1;
            int in_start = out_pos * stride - padding_l;
            for (int k = 0; k < kernel_size; ++k) {
                int in_pos = in_start + k * dilation;
                if (in_pos < 0 || in_pos >= in_length) {
                    continue;
                }
                scalar_t val = input[batch_idx][channel_idx][in_pos];
                if (val > max_val) {
                    max_val = val;
                    max_idx = in_pos;
                }
            }

            output[batch_idx][channel_idx][out_pos] = max_val;
            if (return_indices) {
                indices[batch_idx][channel_idx][out_pos] = max_idx;
            }
        }

        std::vector<torch::Tensor> max_pool1d_forward(
            torch::Tensor input,
            int padding,
            int dilation,
            int kernel_size,
            int stride,
            bool return_indices) {

            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int in_length = input.size(2);

            const int output_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

            auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
            auto output = torch::empty({batch_size, channels, output_length}, options);
            auto indices = torch::empty({batch_size, channels, output_length}, torch::kLong);

            dim3 threads(output_length);
            dim3 blocks(batch_size, channels);

            AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool1d_forward", ([&] {
                max_pool1d_kernel<scalar_t><<<blocks, threads>>>(
                    input.packed_accessor<scalar_t,3>(),
                    output.packed_accessor<scalar_t,3>(),
                    indices.packed_accessor<int64_t,3>(),
                    batch_size,
                    channels,
                    in_length,
                    padding,
                    kernel_size,
                    stride,
                    dilation);
            }));

            return return_indices ? std::vector<torch::Tensor>({output, indices}) : std::vector<torch::Tensor>({output});
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &max_pool1d_forward, "Max Pool1D forward");
        }
        """

        self.max_pool1d = load_inline(
            name="max_pool1d",
            cpp_sources=max_pool1d_source,
            functions="forward",
            verbose=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.max_pool1d.forward(
            x,
            self.padding,
            self.dilation,
            self.kernel_size,
            self.stride,
            self.return_indices
        )
        return outputs[0] if self.return_indices else outputs[0]

def get_inputs():
    x = torch.rand(batch_size, features, sequence_length).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]

batch_size = 64
features = 192
sequence_length = 65536

kernel_size = 8
stride      = 1
padding     = 4
dilation    = 3            
return_indices = False