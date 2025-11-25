import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel combining convolution, scaling, and min reduction
kernel_code = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

#define CUDA_KERNEL_LOOP(i, n)                             \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);      \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

template <typename scalar_t>
__global__ void fused_conv_scale_min(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size,
    int out_height, int out_width, scalar_t scale_factor) {

  // Output dimensions: (batch, 1, out_height, out_width)
  int n = blockIdx.z;
  int out_h = blockIdx.y;
  int out_w = blockIdx.x;
  int c_out = threadIdx.x;

  if (c_out >= out_channels) return;

  // Output index for current thread's position
  int output_offset = ((n * out_height + out_h) * out_width + out_w) * out_channels + c_out;

  // Compute convolution + scale + min
  scalar_t min_val = std::numeric_limits<scalar_t>::max();
  for (int c_in = 0; c_in < in_channels; ++c_in) {
    for (int kh = 0; kh < kernel_size; ++kh) {
      for (int kw = 0; kw < kernel_size; ++kw) {
        int h_in = out_h + kh;
        int w_in = out_w + kw;
        if (h_in < in_height && w_in < in_width) {
          int input_offset = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
          int weight_offset = (c_out * in_channels + c_in) * kernel_size * kernel_size + kh * kernel_size + kw;
          scalar_t val = input[input_offset] * weight[weight_offset];
          val += bias[c_out];  // Add bias before scaling?
          val *= scale_factor;

          // Track minimum across all channels
          if (val < min_val) {
            min_val = val;
          }
        }
      }
    }
  }

  // Write result to output. Only the first channel (dim=1) is kept, so we write min_val to all out_channels?
  // Wait, the output has only 1 channel (as per the original model's min(dim=1, keepdim=True))
  // So all threads per block need to collaborate to find the min across all out_channels
  // Oops! The original code's min is over the channel dimension (dim=1), which is the output of the convolution (before scaling)
  // Wait, original model:
  // x = conv(x) --> shape (batch, out_channels, H', W')
  // x = x * scale --> same shape
  // x = torch.min(x, dim=1, keepdim=True)[0] --> reduces to (batch, 1, H', W')

  // So my mistake earlier: The min is across the original 128 output channels of the convolution, not across the threads here

  // So the previous approach is wrong because each thread here is handling a different output channel (out_channels threads per block)
  // Instead, each spatial position (out_h, out_w) across all out_channels must compute the min across all out_channels

  // Therefore, need to perform a reduction across all out_channels for each spatial position

  // Approach: Each block handles a spatial position (out_h, out_w, n)
  // Each thread in the block handles one out_channel
  // Use warp-based reduction to find the minimum across all channels for that spatial position

  // Let's restructure the kernel:

  // First, compute the value for each channel
  // Then perform a reduction across all channels in the block to find the min

  // Let me rework this part:

  // Each thread computes the convolution+scale for its channel
  // Then, using shared memory, all threads in the block (all channels) collaborate to compute the min

  // Let me adjust the kernel structure.

  // Let's try this again:

  // Shared memory for storing per-channel values
  extern __shared__ scalar_t shared[];

  // Each thread computes its channel's value
  scalar_t val = 0.0;
  for (int c_in = 0; c_in < in_channels; ++c_in) {
    for (int kh = 0; kh < kernel_size; ++kh) {
      for (int kw = 0; kw < kernel_size; ++kw) {
        int h_in = out_h + kh;
        int w_in = out_w + kw;
        if (h_in < in_height && w_in < in_width) {
          int input_offset = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
          int weight_offset = (c_out * in_channels + c_in) * kernel_size * kernel_size + kh * kernel_size + kw;
          val += input[input_offset] * weight[weight_offset];
        }
      }
    }
  }
  val += bias[c_out];
  val *= scale_factor;

  // Store in shared memory
  shared[c_out] = val;

  // Synchronize to ensure all writes are done
  __syncthreads();

  // Now perform reduction in shared memory
  int tid = threadIdx.x;
  for (int s = 1; s < out_channels; s *= 2) {
    if (tid % (2*s) == 0) {
      shared[tid] = min(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }

  // The first thread writes the result
  if (tid == 0) {
    output[((n * out_height + out_h) * out_width + out_w) * 1 + 0] = shared[0];
  }
}

// CPU function to launch the kernel
torch::Tensor fused_conv_scale_min_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    float scale_factor) {

  // Get tensor dimensions
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_height = input.size(2);
  int in_width = input.size(3);
  int out_channels = weight.size(0);  // weight is [out_channels, in_channels, kernel_size, kernel_size]
  int out_height = in_height - kernel_size + 1;  // assuming stride 1, padding 0
  int out_width = in_width - kernel_size + 1;

  // Output has shape (batch_size, 1, out_height, out_width)
  auto output = torch::zeros({batch_size, 1, out_height, out_width}, input.options());

  // Launch configuration
  dim3 threads(out_channels);
  dim3 blocks(out_width, out_height, batch_size);  // blockIdx.x=out_w, blockIdx.y=out_h, blockIdx.z=batch

  // Shared memory size: out_channels elements
  size_t smem_size = out_channels * sizeof(scalar_t);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv_scale_min", ([&] {
    fused_conv_scale_min<scalar_t><<<blocks, threads, smem_size>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size, in_channels, out_channels,
        in_height, in_width, kernel_size,
        out_height, out_width, scale_factor);
  }));

  return output;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_conv_scale_min_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    float scale_factor);
"""

# Compile the fused kernel
fused_conv_kernel = load_inline(
    name='fused_conv_kernel',
    cpp_sources=cpp_source,
    cuda_sources=kernel_code,
    functions=['fused_conv_scale_min_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.scale_factor = scale_factor
        # Initialize convolution parameters
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        # Register the fused kernel
        self.fused_conv = fused_conv_kernel

    def forward(self, x):
        return self.fused_conv.fused_conv_scale_min_cuda(
            x, self.weight, self.bias, kernel_size, self.scale_factor
        )

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]