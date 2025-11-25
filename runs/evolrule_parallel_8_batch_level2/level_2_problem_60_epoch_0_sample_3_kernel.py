// Allocate shared memory for partial sums per thread
__shared__ scalar_t partial_sums[THREADS_PER_BLOCK][2]; // [thread][sum, sum_sq]

// Compute Swish and store partial sums
for (each element in thread's responsibility) {
    // compute swish_val
    partial_sums[threadIdx.x][0] += swish_val;
    partial_sums[threadIdx.x][1] += swish_val * swish_val;
}

// Synchronize, then perform reduction
// Use a loop to combine the partial sums into shared_sums
for (int stride = THREADS_PER_BLOCK/2; stride > 0; stride >>=1) {
    __syncthreads();
    if (threadIdx.x < stride) {
        partial_sums[threadIdx.x][0] += partial_sums[threadIdx.x + stride][0];
        partial_sums[threadIdx.x][1] += partial_sums[threadIdx.x + stride][1];
    }
}
// Then, the first thread stores the total in shared_sums
if (threadIdx.x == 0) {
    shared_sums[0] = partial_sums[0][0];
    shared_sums[1] = partial_sums[0][1];
}