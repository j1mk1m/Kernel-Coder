template <typename scalar_t>
__global__ void group_norm_leaky_relu(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    const float* mean,
    const float* var,
    const float* gamma,
    const float* beta,
    float eps,
    float negative_slope,
    int N, int C, int G) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;

    int c = idx % C;
    int g = c / (C / G);
    int n = idx / C;

    float mu = mean[g];
    float var_val = var[g];
    float denom = rsqrt(var_val + eps);

    scalar_t val = input[idx];
    val = (val - mu) * denom * gamma[c] + beta[c]; // Apply normalization and affine

    // Apply leaky ReLU
    val = (val > 0) ? val : val * negative_slope;
    output[idx] = val;
}