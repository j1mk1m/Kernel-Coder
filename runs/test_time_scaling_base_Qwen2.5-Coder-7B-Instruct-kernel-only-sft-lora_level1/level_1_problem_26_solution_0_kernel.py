c++
// CUDA implementation of GELU
__global__ void gelu_kernel(float* output, const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        float y = 0.5f * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        output[i] = y * x;
    }
}