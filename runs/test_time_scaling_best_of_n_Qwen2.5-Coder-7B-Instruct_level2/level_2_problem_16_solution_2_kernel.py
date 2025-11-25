__device__ float mish(float x) {
    return x * tanhf(logf(1 + expf(x)));
}

__device__ float hardtanh(float x) {
    return fmaxf(-1.0f, fminf(1.0f, x));
}