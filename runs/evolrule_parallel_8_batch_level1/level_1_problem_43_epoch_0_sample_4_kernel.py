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

        # Load the custom CUDA kernel with parameters as macros
        cuda_source = self.generate_cuda_source()
        self.max_pool_3d = load_inline(
            name="max_pool_3d",
            cuda_sources=cuda_source,
            functions=["max_pool_3d_forward"],
            verbose=True
        )

    def generate_cuda_source(self):
        return f"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        #define KERNEL_SIZE {self.kernel_size}
        #define STRIDE {self.stride}
        #define PADDING {self.padding}
        #define DILATION {self.dilation}
        #define RETURN_INDICES {1 if self.return_indices else 0}
        #define CEIL_MODE {1 if self.ceil_mode else 0}

        __global__ void max_pool_3d_forward(
            const float* input,
            float* output,
            int* indices,  // Only used if RETURN_INDICES is true
            int batch_size,
            int channels,
            int in_dim1, int in_dim2, int in_dim3,
            int out_dim1, int out_dim2, int out_dim3
        ) {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * channels * out_dim1 * out_dim2 * out_dim3)
                return;

            int output_idx = idx;
            int d3 = output_idx % out_dim3;
            output_idx /= out_dim3;
            int d2 = output_idx % out_dim2;
            output_idx /= out_dim2;
            int d1 = output_idx % out_dim1;
            output_idx /= out_dim1;
            int channel = output_idx % channels;
            int batch = output_idx / channels;

            // Compute input spatial coordinates with padding
            int in_d1 = d1 * STRIDE - PADDING;
            int in_d2 = d2 * STRIDE - PADDING;
            int in_d3 = d3 * STRIDE - PADDING;

            float max_val = -FLT_MAX;
            int max_idx = -1;
            int input_offset_base = batch * channels * in_dim1 * in_dim2 * in_dim3 +
                                   channel * in_dim1 * in_dim2 * in_dim3;

            for (int k1 = 0; k1 < KERNEL_SIZE; ++k1) {{
                int ker_d1 = in_d1 + k1 * DILATION;
                if (ker_d1 < 0 || ker_d1 >= in_dim1)
                    continue;
                for (int k2 = 0; k2 < KERNEL_SIZE; ++k2) {{
                    int ker_d2 = in_d2 + k2 * DILATION;
                    if (ker_d2 < 0 || ker_d2 >= in_dim2)
                        continue;
                    for (int k3 = 0; k3 < KERNEL_SIZE; ++k3) {{
                        int ker_d3 = in_d3 + k3 * DILATION;
                        if (ker_d3 < 0 || ker_d3 >= in_dim3)
                            continue;

                        int offset = input_offset_base + ker_d1 * in_dim2 * in_dim3 +
                                            ker_d2 * in_dim3 +
                                            ker_d3;
                        float val = input[offset];
                        if (val > max_val) {{
                            max_val = val;
                            max_idx = offset;
                        }}
                    }}
                }}
            }}

            // Compute output storage offsets
            int output_offset = batch * channels * out_dim1 * out_dim2 * out_dim3 +
                                channel * out_dim1 * out_dim2 * out_dim3 +
                                d1 * out_dim2 * out_dim3 +
                                d2 * out_dim3 +
                                d3;

            output[output_offset] = max_val;
            #if RETURN_INDICES
            indices[output_offset] = max_idx;
            #endif
        }}

        torch::Tensor max_pool_3d_forward(torch::Tensor input) {{
            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int in_dim1 = input.size(2);
            const int in_dim2 = input.size(3);
            const int in_dim3 = input.size(4);

            // Calculate output dimensions using ceil_mode
            int compute_dim(int in_dim, int kernel, int stride, int dilation) {{
                int pad_total = 2 * PADDING;
                int kernel_effective = (kernel - 1) * dilation + 1;
                int numerator = in_dim + pad_total - kernel_effective;
                if (CEIL_MODE)
                    return (numerator % stride == 0) ? numerator / stride + 1 : (numerator / stride) + 1;
                else
                    return (numerator / stride) + 1;
            }}

            int out_dim1 = compute_dim(in_dim1, KERNEL_SIZE, STRIDE, DILATION);
            int out_dim2 = compute_dim(in_dim2, KERNEL_SIZE, STRIDE, DILATION);
            int out_dim3 = compute_dim(in_dim3, KERNEL_SIZE, STRIDE, DILATION);

            auto options = input.options();
            auto output = torch::empty({{batch_size, channels, out_dim1, out_dim2, out_dim3}}, options);
            torch::Tensor indices = torch::empty_like(output).to(torch::kInt32);

            const int threads_per_block = 256;
            int num_elements = batch_size * channels * out_dim1 * out_dim2 * out_dim3;
            int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

            max_pool_3d_forward<<<num_blocks, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                #if RETURN_INDICES
                indices.data_ptr<int>(),
                #else
                nullptr,
                #endif
                batch_size, channels, in_dim1, in_dim2, in_dim3,
                out_dim1, out_dim2, out_dim3
            );

            if (RETURN_INDICES) {{
                return std::make_tuple(output, indices);
            }} else {{
                return output;
            }}
        }}
        """

    def forward(self, x):
        result = self.max_pool_3d.max_pool_3d_forward(x)
        if self.return_indices:
            output, indices = result
            return output, indices
        else:
            return result