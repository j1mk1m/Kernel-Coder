#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_avg_pool_kernel(
    const float* input, float* output,
    int batch_size, int channels,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_depth * output_height * output_width) {
        return;
    }

    // Compute indices
    int channel = idx / (output_depth * output_height * output_width);
    int rem = idx % (output_depth * output_height * output_width);
    int d_out = rem / (output_height * output_width);
    int rem_hw = rem % (output_height * output_width);
    int h_out = rem_hw / output_width;
    int w_out = rem_hw % output_width;

    // Compute starting positions in input
    int start_d = d_out * 4;
    int start_h = h_out * 4;
    int start_w = w_out * 4;

    float sum = 0.0;
    for (int dd = 0; dd <4; ++dd) {
        int id = start_d + dd;
        if (id >= input_depth) continue;
        for (int hh =0; hh <4; ++hh) {
            int ih = start_h + hh;
            if (ih >= input_height) continue;
            for (int ww=0; ww <4; ++ww) {
                int iw = start_w + ww;
                if (iw >= input_width) continue;
                // Compute input index
                int input_offset = channel * input_depth * input_height * input_width
                    + id * input_height * input_width
                    + ih * input_width
                    + iw;
                sum += input[input_offset];
            }
        }
    }

    output[idx] = sum / 64.0f;
}

torch::Tensor fused_avg_pool_cuda(torch::Tensor input) {
    // Get input dimensions
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);

    // Compute output dimensions
    auto output_depth = (input_depth -4) /4 +1;
    auto output_height = (input_height -4)/4 +1;
    auto output_width = (input_width -4)/4 +1;

    // Create output tensor
    auto output = torch::empty({batch_size, channels, output_depth, output_height, output_width}, input.options());

    const int num_threads = 1024;
    const int num_blocks = (output.numel() + num_threads -1) / num_threads;

    fused_avg_pool_kernel<<<num_blocks, num_threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width
    );

    return output;
}