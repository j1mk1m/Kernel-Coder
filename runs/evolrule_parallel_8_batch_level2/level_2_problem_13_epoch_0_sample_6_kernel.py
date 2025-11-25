import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

        # Custom fused kernel setup
        fused_kernel_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <cmath>
        #include <algorithm>

        template <typename scalar_t>
        __device__ scalar_t log_sum_exp(scalar_t* values, int size) {
            scalar_t max_val = *std::max_element(values, values + size);
            scalar_t sum = 0;
            for (int i = 0; i < size; ++i) {
                sum += exp(values[i] - max_val);
            }
            return max_val + log(sum);
        }

        __global__ void fused_operations_kernel(
            const torch::PackedTensorAccessor<torch::scalar_type_at<torch::kFloat>,5,torch::RestrictPtrTraits> input,
            torch::PackedTensorAccessor<torch::scalar_type_at<torch::kFloat>,5,torch::RestrictPtrTraits> output,
            const torch::PackedTensorAccessor<torch::scalar_type_at<torch::kFloat>,5,torch::RestrictPtrTraits> bias,
            const int depth_dim,
            const float scaling_factor
        ) {
            int B = input.size(0);
            int C = input.size(1);
            int D = depth_dim;
            int H = input.size(3);
            int W = input.size(4);

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= B*C*H*W) return;

            int w = idx % W;
            int h = (idx / W) % H;
            int c = (idx / (H*W)) % C;
            int b = idx / (C*H*W);

            // Compute mean over depth dimension (D)
            float mean_val = 0;
            for (int d = 0; d < D; ++d) {
                mean_val += input[b][c][d][h][w];
            }
            mean_val /= D;

            // Add bias
            mean_val += bias[0][c][0][0][0];

            // Softmax over channels (C dimension)
            // Use shared memory to handle channel dimension
            extern __shared__ float shared_mem[];
            float* row_data = shared_mem;
            int thread_id = threadIdx.x;

            // Each thread computes one element of the channel row
            row_data[thread_id] = mean_val;
            __syncthreads();

            // Compute log_sum_exp for the channel row
            float log_sum = log_sum_exp<scalar_t>(row_data, C);
            float softmax_val = exp(mean_val - log_sum);

            // Apply tanh and scaling
            softmax_val = tanh(softmax_val) * scaling_factor;

            output[b][c][0][h][w] = softmax_val;
        }

        torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor) {
            auto output = torch::empty_like(input);
            output = output.select(2, 0); // Squeeze depth dimension after mean

            int B = input.size(0);
            int C = input.size(1);
            int H = input.size(3);
            int W = input.size(4);
            int total_elements = B * C * H * W;

            dim3 blocks((total_elements + 255) / 256, 1, 1);
            dim3 threads(256, 1, 1);

            // Shared memory required: C floats per block
            int shared_mem_size = C * sizeof(float);

            fused_operations_kernel<<<blocks, threads, shared_mem_size>>>(
                input.packed_accessor<torch::scalar_type<torch::kFloat>(),5,torch::RestrictPtrTraits>(),
                output.packed_accessor<torch::scalar_type<torch::kFloat>(),5,torch::RestrictPtrTraits>(),
                bias.packed_accessor<torch::scalar_type<torch::kFloat>(),5,torch::RestrictPtrTraits>(),
                input.size(2),
                scaling_factor
            );

            return output;
        }
        """

        # Compile the fused kernel
        fused_ops = load_inline(
            name="fused_ops",
            cpp_sources="",
            cuda_sources=fused_kernel_source,
            functions=["fused_ops_cuda"],
            verbose=True
        )
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias, self.scaling_factor)
        return x