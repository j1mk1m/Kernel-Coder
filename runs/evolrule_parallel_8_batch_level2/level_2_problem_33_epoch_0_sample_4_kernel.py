__global__ void compute_mean_var(
    const float* x_gemm, 
    float* sum_x,
    float* sum_x_sq,
    int batch_size,
    int out_features) 
{
    // Each block handles a feature j
    int j = blockIdx.x;

    if (j >= out_features) return;

    extern __shared__ float shared[];

    int tid = threadIdx.x;

    // Each thread handles an element of the batch
    float local_sum = 0.0f;
    float local_sq = 0.0f;

    for (int i = tid; i < batch_size; i += blockDim.x) {
        float val = x_gemm[i * out_features + j];
        local_sum += val;
        local_sq += val * val;
    }

    // Use shared memory for reduction
    int offset = 0;
    shared[threadIdx.x + offset] = local_sum;
    __syncthreads();

    // Reduce local sums using block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x + offset] += shared[threadIdx.x + s + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum_x[j] = shared[0 + offset];
        sum_x_sq[j] = shared[blockDim.x + offset]; // Wait, need to store both?
    }

    // Wait, this is a problem because we have two values to reduce: sum and sum_sq.

    // Maybe need to handle them separately in shared memory.

    // Alternatively, use two separate shared arrays, or interleave.

    // Let's try using two separate halves of shared memory.

    // The shared memory size needs to be sufficient.

    // Let me adjust the code.

    // Each thread first computes local_sum and local_sq.

    // Store both in shared memory.

    int half_size = blockDim.x;
    shared[threadIdx.x] = local_sum;
    shared[threadIdx.x + half_size] = local_sq;

    __syncthreads();

    // Reduce sum:

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
            shared[threadIdx.x + half_size] += shared[threadIdx.x + s + half_size];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum_x[j] = shared[0];
        sum_x_sq[j] = shared[half_size];
    }
}

// The shared memory size needed is 2 * blockDim.x (since we need space for both sum and sq)
// So, for a block size of 256, shared_mem_size = 256 * 2 * sizeof(float) = 2048 bytes, which is okay.