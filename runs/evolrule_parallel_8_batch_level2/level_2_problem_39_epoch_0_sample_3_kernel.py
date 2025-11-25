__global__ void scale_mul_kernel(
    const float* x,
    const float* scale,
    float* out,
    int batch_size,
    int out_features) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int j = idx % out_features;
        out[idx] = x[idx] * scale[j];
    }
}