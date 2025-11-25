import torch
        import torch.nn as nn
        from torch.utils.cpp_extension import load_inline

        # Define the custom CUDA kernel for global mean
        global_mean_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <int BLOCK_SIZE>
        __global__ void global_mean_kernel(const float* input, float* output, int N, int C, int H, int W) {
            int b = blockIdx.x;
            int c = blockIdx.y;
            if (b >= N || c >= C) return;

            int tid = threadIdx.x;
            float sum = 0.0f;

            for (int idx = tid; idx < H * W; idx += blockDim.x) {
                int h = idx / W;
                int w = idx % W;
                int in_offset = b * C * H * W + c * H * W + h * W + w;
                sum += input[in_offset];
            }

            __shared__ float shared_sum[BLOCK_SIZE];
            shared_sum[tid] = sum;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_sum[tid] += shared_sum[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                float avg = shared_sum[0] / (H * W);
                int out_offset = b * C * 1 * 1 + c * 1 * 1 + 0 * 1 + 0;
                output[out_offset] = avg;
            }
        }

        torch::Tensor global_mean_cuda(torch::Tensor input) {
            const int N = input.size(0);
            const int C = input.size(1);
            const int H = input.size(2);
            const int W = input.size(3);

            auto output = torch::empty({N, C, 1, 1}, input.options());

            const int block_size = 256;
            dim3 blocks(N, C);
            dim3 threads(block_size);

            global_mean_kernel<block_size><<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W);

            return output;
        }
        """

        global_mean_cpp = """
        torch::Tensor global_mean_cuda(torch::Tensor input);
        """

        # Compile the inline CUDA code for global mean
        global_mean = load_inline(
            name="global_mean",
            cpp_sources=global_mean_cpp,
            cuda_sources=global_mean_source,
            functions=["global_mean_cuda"],
            verbose=True,
            extra_cflags=[""],
            extra_ldflags=[""],
        )

        class ModelNew(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
                super().__init__()
                self.conv_transpose = nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
                )
                self.multiplier = multiplier  # stored as a buffer
                self.register_buffer('multiplier_buffer', torch.tensor([self.multiplier], dtype=torch.float32))

            def forward(self, x):
                x = self.conv_transpose(x)
                x = global_mean.global_mean_cuda(x)
                x = x * self.multiplier_buffer  # use the buffer to ensure tensor type
                return x

        def get_inputs():
            return [torch.rand(batch_size, in_channels, height, width)]

        def get_init_inputs():
            return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]