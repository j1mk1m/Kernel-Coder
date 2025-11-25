import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

        avg_pool_3d_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void avg_pool3d_kernel(const scalar_t* input, scalar_t* output,
            int batch_size, int channels, int in_depth, int in_height, int in_width,
            int kernel_size, int stride, int padding) {

            const int out_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
            const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
            const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

            const int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx >= batch_size * channels * out_depth * out_height * out_width) return;

            const int w = idx % out_width;
            const int h = (idx / out_width) % out_height;
            const int d = (idx / (out_width * out_height)) % out_depth;
            const int c = (idx / (out_width * out_height * out_depth)) % channels;
            const int b = idx / (out_width * out_height * out_depth * channels);

            const int in_d_start = d * stride - padding;
            const int in_h_start = h * stride - padding;
            const int in_w_start = w * stride - padding;

            scalar_t sum = 0.0;
            int count = 0;

            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        const int id = in_d_start + kd;
                        const int ih = in_h_start + kh;
                        const int iw = in_w_start + kw;

                        if (id >= 0 && id < in_depth &&
                            ih >= 0 && ih < in_height &&
                            iw >= 0 && iw < in_width) {
                            sum += input[b * channels * in_depth * in_height * in_width +
                                        c * in_depth * in_height * in_width +
                                        id * in_height * in_width +
                                        ih * in_width + iw];
                            count++;
                        }
                    }
                }
            }

            output[idx] = sum / static_cast<scalar_t>(count);
        }

        torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int in_depth = input.size(2);
            const int in_height = input.size(3);
            const int in_width = input.size(4);

            const int out_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
            const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
            const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

            auto output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, input.options());

            const int total_threads = batch_size * channels * out_depth * out_height * out_width;
            const int block_size = 256;
            const int num_blocks = (total_threads + block_size - 1) / block_size;

            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool3d_cuda", ([&] {
                avg_pool3d_kernel<scalar_t><<<num_blocks, block_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels, in_depth, in_height, in_width,
                    kernel_size, stride, padding);
            }));

            return output;
        }
        """

        avg_pool_3d_cpp_source = (
            "torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"
        )

        self.avg_pool_cuda = load_inline(
            name="avg_pool_cuda",
            cpp_sources=avg_pool_3d_cpp_source,
            cuda_sources=avg_pool_3d_source,
            functions=["avg_pool3d_cuda"],
            verbose=True,
            extra_cflags=[""],
            extra_ldflags=[""],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_cuda.avg_pool3d_cuda(
            x, self.kernel_size, self.stride, self.padding
        )