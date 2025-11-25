__global__ void fused_mean_add_bias_kernel(
    const float* input,
    const float* bias,
    float* output,
    int B, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H * W) return;

    int b = idx / (C * H * W);
    int rem1 = idx % (C * H * W);
    int c = rem1 / (H * W);
    int rem2 = rem1 % (H * W);
    int h = rem2 / W;
    int w = rem2 % W;

    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
        int input_offset = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        sum += input[input_offset];
    }
    sum /= static_cast<float>(D);
    sum += bias[c]; // bias is per channel (C elements)
    
    int output_offset = b * C * H * W + c * H * W + h * W + w; // since depth is 1
    output[output_offset] = sum;
}