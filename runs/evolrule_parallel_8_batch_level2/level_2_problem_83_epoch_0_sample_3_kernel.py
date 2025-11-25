#include <curand_kernel.h>

__global__ void dropout_kernel(float* data, int size, float p, float scale, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);
        float rand = curand_uniform(&state);
        if (rand < p) {
            data[idx] = 0;
        } else {
            data[idx] *= scale;
        }
    }
}