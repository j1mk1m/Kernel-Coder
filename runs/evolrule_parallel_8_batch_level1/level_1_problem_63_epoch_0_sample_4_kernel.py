import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize weights and bias like PyTorch's Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Define and load the CUDA kernels
        from torch.utils.cpp_extension import load_inline

        custom_conv2d_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cublas_v2.h>

        __global__ void im2col_kernel(const float* input, float* col, int N, int C, int H, int W,
                                     int K, int stride, int padding, int dilation,
                                     int H_out, int W_out) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= C * K * K * N * H_out * W_out) return;

            int col_idx = idx % (N * H_out * W_out);
            int row_idx = idx / (N * H_out * W_out);

            int c_in = row_idx / (K * K);
            int kh = (row_idx / K) % K;
            int kw = row_idx % K;

            int n = col_idx / (H_out * W_out);
            int pos = col_idx % (H_out * W_out);
            int h_out = pos / W_out;
            int w_out = pos % W_out;

            int h_in = h_out * stride + kh * dilation - padding;
            int w_in = w_out * stride + kw * dilation - padding;

            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) {
                col[idx] = 0.0f;
                return;
            }

            int input_offset = n * C * H * W + c_in * H * W + h_in * W + w_in;
            col[idx] = input[input_offset];
        }

        torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                        int stride, int padding, int dilation, int groups) {
            auto device = input.device();
            auto stream = at::cuda::getCurrentCUDAStream();

            int N = input.size(0);
            int C_in = input.size(1);
            int H_in = input.size(2);
            int W_in = input.size(3);
            int C_out = weight.size(0);
            int K = weight.size(2);

            // Handle groups here if needed (currently only supports groups=1)
            if (groups != 1) {
                throw std::runtime_error("Groups >1 not supported in this example.");
            }

            int H_out = (H_in + 2 * padding - dilation * (K - 1)) / stride + 1;
            int W_out = (W_in + 2 * padding - dilation * (K - 1)) / stride + 1;

            int num_cols = N * H_out * W_out;
            int num_rows = C_in * K * K;
            int col_size = num_rows * num_cols;

            auto col = torch::empty({num_rows, num_cols}, torch::dtype(input.dtype()).device(device));

            dim3 threads(256);
            dim3 blocks((col_size + threads.x - 1) / threads.x);

            im2col_kernel<<<blocks, threads, 0, stream>>>(input.data_ptr<float>(),
                                                        col.data_ptr<float>(),
                                                        N, C_in, H_in, W_in,
                                                        K, stride, padding, dilation,
                                                        H_out, W_out);

            auto weight_reshaped = weight.view({C_out, C_in * K * K});

            cublasHandle_t handle;
            cublasCreate(&handle);

            const float alpha = 1.0f;
            const float beta = 0.0f;
            float* d_weight = weight_reshaped.data_ptr<float>();
            float* d_col = col.data_ptr<float>();

            auto output_mat = torch::empty({C_out, num_cols}, torch::dtype(input.dtype()).device(device));
            float* d_out = output_mat.data_ptr<float>();

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        C_out, num_cols, num_rows,
                        &alpha, d_weight, C_out,
                        d_col, num_rows,
                        &beta, d_out, C_out);

            cublasDestroy(handle);

            auto output = output_mat.view({N, C_out, H_out, W_out});

            if (bias.defined()) {
                output = output + bias.view({1, -1, 1, 1});
            }

            return output;
        }
        """
        custom_conv2d = load_inline(
            name="custom_conv2d",
            cuda_sources=custom_conv2d_source,
            functions=["custom_conv2d_cuda"],
            verbose=True,
            extra_cflags=["-std=c++14"],
            extra_cuda_cflags=["-std=c++14"]
        )
        self.custom_conv2d = custom_conv2d

    def forward(self, x):
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return self.custom_conv2d.custom_conv2d_cuda(
            x,
            self.weight,
            bias_tensor,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )