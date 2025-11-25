template <int GROUP_SIZE>
__global__ void fused_kernel(
    const float* input, const float* weight, const float* bias,
    const float* gamma, const float* beta, const float* multiply_weight,
    float* output,
    int batch_size, int in_features, int out_features, int num_groups) {

    int b = blockIdx.x;
    int g = blockIdx.y;
    int tid = threadIdx.x;

    // Ensure that group_size is correct (should be out_features / num_groups)
    const int group_size = GROUP_SIZE;

    // Each thread in the block processes one channel in the group
    int c = g * group_size + tid;

    // Shared memory for input and intermediate values
    extern __shared__ float s_data[];

    float* s_input = s_data;
    float* s_intermediate = s_input + in_features;

    // Load the input vector into shared memory
    for (int i = tid; i < in_features; i += blockDim.x) {
        s_input[i] = input[b * in_features + i];
    }
    __syncthreads();

    // Compute GEMM for this channel
    float val = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        val += weight[c * in_features + k] * s_input[k];
    }
    val += bias[c];

    // Store intermediate value in shared memory
    s_intermediate[tid] = val;
    __syncthreads();

    // Compute group mean and variance
    float sum_val = 0.0f;
    for (int i = 0; i < group_size; ++i) {
        sum_val += s_intermediate[i];
    }
    __shared__ float local_sum;
    if (tid == 0) {
        local_sum = sum_val;
    }
    __syncthreads();
    float mean = local_sum / group_size;

    float sum_sq = 0.0f;
    for (int i = 0; i < group_size; ++i) {
        float diff = s_intermediate[i] - mean;
        sum_sq += diff * diff;
    }
    __shared__ float local_sum_sq;
    if (tid == 0) {
        local_sum_sq = sum_sq;
    }
    __syncthreads();
    float variance = local_sum_sq / group_size + 1e-5; // epsilon
    float inv_std = rsqrt(variance);

    // Compute normalized value
    float x = s_intermediate[tid];
    float normalized = (x - mean) * inv_std;

    // Apply gamma and beta
    normalized = normalized * gamma[c] + beta[c];

    // First Swish: x * sigmoid(x)
    float swish1 = normalized * (1.0f / (1.0f + expf(-normalized)));

    // Multiply by multiply_weight
    float multiplied = swish1 * multiply_weight[c];

    // Second Swish
    float swish2 = multiplied * (1.0f / (1.0f + expf(-multiplied)));

    // Write output
    output[b * out_features + c] = swish2;
}