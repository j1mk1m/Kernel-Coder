import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class FusedBatchNorm2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var, eps):
        ctx.save_for_backward(input, weight, running_mean, running_var)
        ctx.eps = eps
        return batchnorm_forward_cuda(input, weight, bias, running_mean, running_var, eps)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, running_mean, running_var = ctx.saved_tensors
        eps = ctx.eps
        grad_input, grad_weight, grad_bias = batchnorm_backward_cuda(
            grad_output, input, weight, running_mean, running_var, eps
        )
        return grad_input, grad_weight, grad_bias, None, None, None

class ModelNew(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5

        # Initialize weights and bias similar to PyTorch's BatchNorm2d
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        # Load the CUDA kernels
        self.load_cuda_kernels()

    def load_cuda_kernels(self):
        kernel_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <vector>

        template <typename scalar_t>
        __global__ void batchnorm_forward_kernel(
            const scalar_t* __restrict__ input,
            const scalar_t* __restrict__ weight,
            const scalar_t* __restrict__ bias,
            const scalar_t* __restrict__ running_mean,
            const scalar_t* __restrict__ running_var,
            scalar_t* __restrict__ output,
            const int N,
            const int C,
            const int spatial,
            const float eps
        ) {
            const int total_elements = N * C * spatial;
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_elements) return;

            const int in_batch_idx = idx % (C * spatial);
            const int c = in_batch_idx / spatial;
            const int s = in_batch_idx % spatial;

            scalar_t mean = running_mean[c];
            scalar_t var = running_var[c];
            scalar_t inv_std = 1.0f / sqrt(var + eps);

            scalar_t x = input[idx];
            output[idx] = (x - mean) * inv_std * weight[c] + bias[c];
        }

        at::Tensor batchnorm_forward_cuda(
            const at::Tensor input,
            const at::Tensor weight,
            const at::Tensor bias,
            const at::Tensor running_mean,
            const at::Tensor running_var,
            const float eps
        ) {
            const auto N = input.size(0);
            const auto C = input.size(1);
            const auto spatial = input.size(2) * input.size(3);
            const int total_elements = N * C * spatial;

            auto input_contig = input.contiguous();
            auto output = at::empty_like(input_contig);

            const int threads = 256;
            const int blocks = (total_elements + threads - 1) / threads;

            AT_DISPATCH_FLOATING_TYPES(input.type(), "batchnorm_forward_cuda", ([&] {
                batchnorm_forward_kernel<scalar_t><<<blocks, threads>>>(
                    input_contig.data<scalar_t>(),
                    weight.data<scalar_t>(),
                    bias.data<scalar_t>(),
                    running_mean.data<scalar_t>(),
                    running_var.data<scalar_t>(),
                    output.data<scalar_t>(),
                    N, C, spatial, eps
                );
            }));

            cudaDeviceSynchronize();
            return output;
        }

        template <typename scalar_t>
        __global__ void batchnorm_backward_kernel(
            const scalar_t* __restrict__ grad_output,
            const scalar_t* __restrict__ input,
            const scalar_t* __restrict__ weight,
            const scalar_t* __restrict__ running_mean,
            const scalar_t* __restrict__ running_var,
            scalar_t* __restrict__ grad_input,
            scalar_t* __restrict__ grad_weight,
            scalar_t* __restrict__ grad_bias,
            const int N,
            const int C,
            const int spatial,
            const float eps
        ) {
            const int total_elements = N * C * spatial;
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_elements) return;

            const int in_batch_idx = idx % (C * spatial);
            const int c = in_batch_idx / spatial;
            const int s = in_batch_idx % spatial;

            scalar_t mean = running_mean[c];
            scalar_t var = running_var[c];
            scalar_t inv_std = 1.0f / sqrt(var + eps);

            scalar_t x_hat = (input[idx] - mean) * inv_std;
            atomicAdd(grad_weight + c, grad_output[idx] * x_hat);
            atomicAdd(grad_bias + c, grad_output[idx]);

            grad_input[idx] = grad_output[idx] * weight[c] * inv_std;
        }

        std::tuple<at::Tensor, at::Tensor, at::Tensor> batchnorm_backward_cuda(
            const at::Tensor grad_output,
            const at::Tensor input,
            const at::Tensor weight,
            const at::Tensor running_mean,
            const at::Tensor running_var,
            const float eps
        ) {
            const auto N = input.size(0);
            const auto C = input.size(1);
            const auto spatial = input.size(2) * input.size(3);
            const int total_elements = N * C * spatial;

            auto grad_output_contig = grad_output.contiguous();
            auto input_contig = input.contiguous();

            auto grad_input = at::zeros_like(input_contig);
            auto grad_weight = at::zeros_like(weight);
            auto grad_bias = at::zeros_like(weight);

            const int threads = 256;
            const int blocks = (total_elements + threads - 1) / threads;

            AT_DISPATCH_FLOATING_TYPES(input.type(), "batchnorm_backward_cuda", ([&] {
                batchnorm_backward_kernel<scalar_t><<<blocks, threads>>>(
                    grad_output_contig.data<scalar_t>(),
                    input_contig.data<scalar_t>(),
                    weight.data<scalar_t>(),
                    running_mean.data<scalar_t>(),
                    running_var.data<scalar_t>(),
                    grad_input.data<scalar_t>(),
                    grad_weight.data<scalar_t>(),
                    grad_bias.data<scalar_t>(),
                    N, C, spatial, eps
                );
            }));

            cudaDeviceSynchronize();

            grad_weight /= N;
            grad_bias /= N;

            return std::make_tuple(grad_input, grad_weight, grad_bias);
        }
        """
        cuda_module = load_inline(
            name='fused_batchnorm',
            cuda_sources=kernel_source,
            functions=[
                'at::Tensor batchnorm_forward_cuda',
                'std::tuple<at::Tensor, at::Tensor, at::Tensor> batchnorm_backward_cuda'
            ],
            extra_cuda_cflags=['-arch=sm_80'],
            verbose=True
        )
        global batchnorm_forward_cuda, batchnorm_backward_cuda
        batchnorm_forward_cuda = cuda_module.batchnorm_forward_cuda
        batchnorm_backward_cuda = cuda_module.batchnorm_backward_cuda

    def forward(self, x):
        return FusedBatchNorm2dFunction.apply(
            x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
        )

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]