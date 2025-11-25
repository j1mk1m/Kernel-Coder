__global__ void riemann_siegel_x_kernel(const float* input, float s, float* output, int elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= elements) return;

    output[idx] = riemann_siegel_x(s, input[idx]);
}