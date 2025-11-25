template <typename scalar_t>
__global__ void grad_m_kernel(
    const torch::PackedTensorAccessor<scalar_t,5> x_in,
    const torch::PackedTensorAccessor<scalar_t,4> multiplier,
    const torch::PackedTensorAccessor<scalar_t,5> grad_output,
    torch::PackedTensorAccessor<scalar_t,4> grad_m,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N*C*D*H*W) return;

    int n = idx / (C*D*H*W);
    int c = (idx / (D*H*W)) % C;
    int d = (idx / (H*W)) % D;
    int h = (idx / W) % H;
    int w = idx % W;

    // Compute y1
    scalar_t x_val = x_in[n][c][d][h][w];
    scalar_t y1 = (x_val > 0) ? x_val : 0.2 * x_val;

    // Compute y2
    scalar_t m_val = multiplier[c][0][0][0];
    scalar_t y2 = y1 * m_val;

    // Compute grad_y2 = grad_output * dy3/dy2
    scalar_t grad_out_val = grad_output[n][c][d][h][w];
    scalar_t dy3_dy2 = (y2 > 0) ? 1.0 : 0.2;
    scalar_t grad_y2 = grad_out_val * dy3_dy2;

    // Atomic add to grad_m[c][0][0][0]
    atomicAdd(&grad_m[c][0][0][0], grad_y2 * y1);
}