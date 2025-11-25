__global__ void linear_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features) {
    int batch = blockIdx.x;
    int out = threadIdx.x;

    if (out >= out_features) return;

    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        sum += input[batch * in_features + i] * weight[out * in_features + i];
    }
    output[batch * out_features + out] = sum + bias[out];
}