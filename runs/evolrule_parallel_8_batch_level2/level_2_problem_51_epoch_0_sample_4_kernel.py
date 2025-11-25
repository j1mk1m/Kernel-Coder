import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Gemm (Linear) and Subtract
gemm_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void fused_linear_subtract_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ subtract_param,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    scalar_t sum[out_features];
    #pragma unroll
    for (int i = 0; i < out_features; i++) {
        sum[i] = (bias ? bias[i] : 0);
    }

    for (int j = 0; j < in_features; j++) {
        scalar_t input_val = input[idx * in_features + j];
        for (int i = 0; i < out_features; i++) {
            sum[i] += input_val * weight[j * out_features + i];
        }
    }

    for (int i = 0; i < out_features; i++) {
        output[idx * out_features + i] = sum[i] - subtract_param[i];
    }
}

torch::Tensor fused_linear_subtract_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor subtract_param) {

    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(1);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_linear_subtract_cuda", ([&] {
        fused_linear_subtract_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            (bias.defined() ? bias.data_ptr<scalar_t>() : nullptr),
            subtract_param.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features);
    }));

    return output;
}
"""

# Custom CUDA kernel for fused GlobalAvgPool and LogSumExp
globalavgpool_lse_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void fused_globalavgpool_lse_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int features) {

    // GlobalAvgPool (mean over features)
    // Then LogSumExp over the same dimension

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < batch_size; b += blockDim.x * gridDim.x) {
        scalar_t sum = 0;
        scalar_t max_val = -FLT_MAX;
        for (int f = 0; f < features; f++) {
            scalar_t val = input[b * features + f];
            sum += val;
            if (val > max_val) max_val = val;
        }
        scalar_t mean = sum / features;

        // Compute LogSumExp of the mean (since it's a scalar per batch)
        // Wait, original code applies LogSumExp after GlobalAvgPool (which reduces to 1 feature)
        // Wait looking back:
        // Original model after GlobalAvgPool: x has shape (batch, 1)
        // Then logsumexp over dim=1 (so over the singleton dimension?), but that would collapse to scalar.
        // Wait, the original model's GlobalAvgPool is over dim=1 (features), resulting in (batch, 1)
        // Then logsumexp over dim=1 would collapse to (batch,). But the residual add requires adding to original_x (shape batch x in_features)
        // Wait the original code's residual add has a problem? Because after processing, x is (batch,1), and original_x is (batch, in_features)
        // The code adds them, which would require broadcasting, but the shapes don't align. Wait the original model's forward has:

        // The original model:
        // original_x = x.clone().detach()  # shape (batch, in_features)
        // After Gemm, Subtract, GlobalAvgPool (dim=1) becomes (batch, 1)
        // LogSumExp over dim=1 (so reduces to (batch,)), but then GELU is applied, then residual add with original_x (shape batch x in_features)
        // Wait this would cause a shape mismatch. Wait let me check:

        // Let me recheck the original code:

        original_x = x.clone().detach()  # Let's say initial x is (batch, in_features)
        x = self.gemm(x)  # now (batch, out_features)
        x = x - self.subtract  # still (batch, out_features)
        x = torch.mean(x, dim=1, keepdim=True)  # becomes (batch, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True)  # Now x is (batch, 1), since logsumexp over dim=1 (the singleton dim) would sum to scalar per batch, but keepdim=True gives (batch, 1)
        x = F.gelu(x)  # still (batch, 1)
        x = x + original_x  # original_x is (batch, in_features). Adding (batch,1) + (batch, in_features) via broadcasting would result in (batch, in_features)
        # So the residual add is valid via broadcasting

        // So the GlobalAvgPool reduces to (batch,1), then LogSumExp over dim=1 (the features dimension which is size 1) does nothing except compute log(exp(mean)), since sum(exp(mean)) is exp(mean). So log(exp(mean)) is mean. So the LogSumExp here is redundant? Wait no.

        Wait, let me compute logsumexp of a single element. Suppose the input to logsumexp is a tensor of shape (batch,1). Then for each batch element, logsumexp over dim=1 would compute log(exp(x_{b,0})). Since there's only one element, this is x_{b,0}. So effectively, the LogSumExp here is a no-op? That can't be. So maybe there's a mistake in the original model?

        Wait the original model's code:

        After GlobalAvgPool (mean over dim=1), x is (batch, 1). Then logsumexp over dim=1 (the same dimension). Since dim=1 has size 1, logsumexp is equivalent to the single element. So that operation is redundant. Maybe it's a mistake in the original model. But since we need to preserve the original behavior, we must replicate it even if it's redundant.

        Therefore, in the fused kernel, after computing the mean (GlobalAvgPool), we then compute logsumexp over that dimension (which is size 1), which just returns the same value. So the fused operation here is just the GlobalAvgPool, but we need to do both steps as per the original code.

        So in the kernel:

        For each sample, compute mean over features (GlobalAvgPool), then compute logsumexp over the same dimension (which is now size 1). The result is same as the mean.

        So effectively, the LogSumExp here is redundant. But to match the original code, we have to perform it.

        So the fused kernel can just compute the GlobalAvgPool, and then the LogSumExp (which does nothing), so the output is same as GlobalAvgPool.

        Therefore, the fused kernel can just compute the GlobalAvgPool (mean) and then output that. But we need to do both steps.

        Anyway, let's proceed.

        So here, after GlobalAvgPool (mean over features), the tensor is (batch, 1). Then logsumexp over dim=1 (the features dimension) which has size 1, so it's equivalent to the single element. So the LogSumExp is redundant. But since the original code does it, we must preserve it.

        Therefore, the fused kernel for GlobalAvgPool and LogSumExp can be written as:

        Compute mean over features (GlobalAvgPool), then compute logsumexp over the same dimension (which is size 1). The result is the same as the mean.

        So the kernel can compute the mean and return that, but we have to do the steps to match the original.

        Alternatively, perhaps there's a mistake in the original model, but we have to replicate exactly.

        So proceeding with the code:

        Compute the mean (GlobalAvgPool), then compute logsumexp over dim=1 (the features dimension which is now 1).

        For logsumexp computation:

        max_val = max over elements in the dimension (here, only one element)
        sum_exp = sum(exp(x_i - max_val))
        log(sum_exp) + max_val

        Since there's only one element, this is just x_i.

        So the logsumexp here is redundant, but we have to do it.

        So the kernel can proceed as:

        For each batch element:

            Compute mean of features (GlobalAvgPool)
            Then compute logsumexp over the same dimension (which is now size 1). Since there's only one element, it's the same as the mean.

            So the output is just the mean.

        So the fused kernel can just compute the mean (GlobalAvgPool), then return it (since logsumexp over dim=1 of a single element is the same as the element).

        Therefore, the kernel can just compute the mean.

        But to strictly follow the original code, even if redundant, we need to perform both steps.

        Anyway, proceeding with the code:

        For each batch, compute the mean (GlobalAvgPool), then compute logsumexp over dim=1 (which is the same as the mean).

        Thus, the output after both operations is the same as the GlobalAvgPool's output.

        Therefore, the kernel can compute the mean and return it as the result of both operations.

        So the kernel code:

        For each batch:

            Compute the sum over features, divide by features (mean)
            Then compute logsumexp over dim=1 (which is the same as the mean)

        So the code:

        sum_val = sum over features (input[b * features + f])
        mean_val = sum_val / features

        max_val = mean_val (since only one element)
        sum_exp = exp(mean_val - max_val) = 1
        log_sum_exp = log(1) + max_val = mean_val

        So the result is mean_val.

        Therefore, the kernel can just compute the mean and return that.

        So the fused kernel can just compute the mean (GlobalAvgPool), which is the same as the result of both operations.

        Thus, the kernel can proceed as follows:

        for each batch:
            compute mean over features, store in output.

        So the code:

        scalar_t sum = 0;
        for (int f = 0; f < features; f++) {
            sum += input[b * features + f];
        }
        output[b * 1] = sum / features;

        But since we have to process in parallel, let's use a block per batch.

        So the kernel code:

        __global__ void fused_globalavgpool_lse_kernel(...) {
            for (b in batches) {
                compute sum over features, divide by features, store in output[b]
            }
        }

        So the kernel will process each batch element in parallel.

        Let's write the code accordingly:

        So the kernel:

        template<typename scalar_t>
        __global__ void fused_globalavgpool_lse_kernel(
            const scalar_t* input,
            scalar_t* output,
            int batch_size,
            int features) {

            int b = blockIdx.x * blockDim.x + threadIdx.x;
            if (b >= batch_size) return;

            scalar_t sum = 0;
            for (int f = 0; f < features; f++) {
                sum += input[b * features + f];
            }
            output[b] = sum / features;
        }

        The output is stored in a 1D tensor? Or keepdim=True?

        Wait the original code's GlobalAvgPool uses keepdim=True, resulting in (batch, 1), then LogSumExp also with keepdim=True would keep it as (batch,1).

        So the output should be (batch,1).

        Therefore, the output tensor in the kernel should have size (batch,1).

        So the kernel's output is stored as output[b * 1], but the output tensor is of shape (batch_size, 1).

        So the code:

        output[b * 1] = ... 

        But in terms of memory layout, for a (batch_size, 1) tensor, the offset for batch b is b * 1.

        So the kernel is okay.

        Then the kernel launch:

        The grid and block configuration can be 1D.

        Let's set threads per block as 256.

        const int threads = 256;
        const int blocks = (batch_size + threads - 1) / threads;

        kernel<<<blocks, threads>>>(...)

        So the kernel code is as above.

        Thus, the fused_globalavgpool_lse_cuda function can be written as:

        torch::Tensor fused_globalavgpool_lse_cuda(torch::Tensor input) {
            int batch_size = input.size(0);
            int features = input.size(1);

            auto output = torch::empty({batch_size, 1}, input.options());

            const int threads = 256;
            const int blocks = (batch_size + threads - 1) / threads;

            AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_globalavgpool_lse_cuda", ([&] {
                fused_globalavgpool_lse_kernel<scalar_t><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    features);
            }));

            return output;
        }
    """
# Wait but the LogSumExp requires processing the dim=1 which is already size 1, so as above, it's redundant. Thus, the fused kernel just does GlobalAvgPool.

# However, in the original model, the GlobalAvgPool is followed by LogSumExp(dim=1, keepdim=True), which would take the (batch,1) tensor and apply logsumexp over the dim=1 (the singleton dimension), which produces the same as the original value. So the output after both operations is the same as the GlobalAvgPool's output. So the fused kernel can just compute the GlobalAvgPool.

# Therefore, the code above is correct.

# Custom CUDA kernel for GELU (approximation)
# Note: GELU can be approximated with a tanh-based formula for faster computation.
gelu_source = """
#include <torch/extension.h>
#include <math.h>

template <typename scalar_t>
__global__ void gelu_cuda_forward(const scalar_t* input, scalar_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t x = input[idx];
        scalar_t tanh_out = tanh((sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
        output[idx] = 0.5 * x * (1 + tanh_out);
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "gelu_cuda_forward", ([&] {
        gelu_cuda_forward<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}
"""

# Custom CUDA kernel for ResidualAdd with broadcasting
# The residual add is x (shape batch x 1) + original_x (shape batch x in_features)
# The output will be batch x in_features, with each element of x's row added to all elements of original_x's row
residual_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void residual_add_kernel(
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ original,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_features) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_features) return;

    int b = idx / in_features;
    int f = idx % in_features;

    output[idx] = original[idx] + residual[b];
}

torch::Tensor residual_add_cuda(
    torch::Tensor residual,
    torch::Tensor original) {

    int batch_size = residual.size(0);
    int in_features = original.size(1);

    auto output = torch::empty_like(original);

    const int threads = 256;
    const int blocks = (batch_size * in_features + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(original.type(), "residual_add_cuda", ([&] {
        residual_add_kernel<scalar_t><<<blocks, threads>>>(
            residual.data_ptr<scalar_t>(),
            original.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features);
    }));

    return output;
}
"""

# Compile the CUDA kernels
fused_linear_subtract = load_inline(
    name="fused_linear_subtract",
    cuda_sources=gemm_subtract_source,
    functions=["fused_linear_subtract_cuda"],
    verbose=True
)

fused_globalavgpool_lse = load_inline(
    name="fused_globalavgpool_lse",
    cuda_sources=globalavgpool_lse_source,
    functions=["fused_globalavgpool_lse_cuda"],
    verbose=True
)

gelu_cuda_op = load_inline(
    name="gelu_cuda",
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True
)

residual_add_cuda_op = load_inline(
    name="residual_add_cuda",
    cuda_sources=residual_add_source,
    functions=["residual_add_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features).T)  # Transposed to match the kernel's expectation (input * weight)
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.subtract = nn.Parameter(torch.randn(out_features))
        # The other operations are handled via CUDA kernels

    def forward(self, x):
        original_x = x.clone().detach()

        # Fused Gemm (Linear) and Subtract
        x = fused_linear_subtract.fused_linear_subtract_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.empty(0),
            self.subtract
        )

        # Fused GlobalAvgPool and LogSumExp
        x = fused_globalavgpool_lse.fused_globalavgpool_lse_cuda(x)

        # GELU (custom kernel)
        x = gelu_cuda_op.gelu_cuda(x)

        # ResidualAdd with broadcasting (custom kernel)
        x = residual_add_cuda_op.residual_add_cuda(x, original_x)

        return x

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]