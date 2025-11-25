__global__ void scale_kernel(float* out, const float* in, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] * scale;
    }
}