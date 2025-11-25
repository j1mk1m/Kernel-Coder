maxpool_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void maxpool_clamp_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int channels,
    const int batch_size,
    const float min_val,
    const float max_val
) {
    // Each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_depth * output_height * output_width) {
        return;
    }
    
    // Calculate output indices
    int output_w = idx % output_width;
    int tmp = idx / output_width;
    int output_h = tmp % output_height;
    tmp /= output_height;
    int output_d = tmp % output_depth;
    tmp /= output_depth;
    int c = tmp % channels;
    int n = tmp / channels;
    
    // Calculate input indices
    int input_start_d = output_d * 2;
    int input_start_h = output_h * 2;
    int input_start_w = output_w * 2;
    
    scalar_t max_val = -INFINITY;
    for (int dz = 0; dz < 2; dz++) {
        int id = input_start_d + dz;
        if (id >= input_depth) continue;
        for (int hy = 0; hy < 2; hy++) {
            int ih = input_start_h + hy;
            if (ih >= input_height) continue;
            for (int wx = 0; wx < 2; wx++) {
                int iw = input_start_w + wx;
                if (iw >= input_width) continue;
                int in_idx = n * channels * input_depth * input_height * input_width
                            + c * input_depth * input_height * input_width
                            + id * input_height * input_width
                            + ih * input_width
                            + iw;
                scalar_t val = input[in_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    
    // Apply clamp
    scalar_t clamped_val = max_val < min_val ? min_val : (max_val > max_val ? max_val : max_val);
    output[idx] = clamped_val;
}

at::Tensor maxpool_clamp_cuda(at::Tensor input, float min_val, float max_val) {
    // Get input dimensions
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int output_depth = (input_depth + 1) / 2;
    int output_height = (input_height + 1) / 2;
    int output_width = (input_width + 1) / 2;
    
    auto output = at::empty({batch_size, channels, output_depth, output_height, output_width}, input.options());
    
    int threads = 256;
    int num_elements = batch_size * channels * output_depth * output_height * output_width;
    int blocks = (num_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "maxpool_clamp_cuda", ([&] {
        maxpool_clamp_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            input_depth, input_height, input_width,
            output_depth, output_height, output_width,
            channels, batch_size,
            min_val, max_val
        );
    }));
    
    return output;
}
"""

maxpool_clamp_cpp_source = "at::Tensor maxpool_clamp_cuda(at::Tensor input, float min_val, float max_val);"

maxpool_clamp = load_inline(
    name="maxpool_clamp",
    cpp_sources=maxpool_clamp_cpp_source,
    cuda_sources=maxpool_clamp_source,
    functions=["maxpool_clamp_cuda"],
    verbose=True
)