__global__ void layer_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int B, int C, int D, int H, int W,
    float eps) {

    // Compute the block's spatial position (b, d, h, w)
    int b = blockIdx.x;
    int d = blockIdx.y;
    int h = (blockIdx.z / W);
    int w = (blockIdx.z % W);

    // Thread index corresponds to channel c
    int c = threadIdx.x;

    // Shared memory for storing input values and partial sums
    __shared__ float s_data[64]; // assuming C is 64, but should be a template or constant.

    // Load the value for this channel and spatial position
    float val = input[get_linear_index(b, c, d, h, w, C, D, H, W)];
    s_data[c] = val;

    __syncthreads();

    // Compute sum and sum_squares using parallel reduction
    float sum = 0.0f;
    float sum_squares = 0.0f;
    for (int i = 0; i < C; i += blockDim.x) {
        if (i + threadIdx.x < C) {
            sum += s_data[i + threadIdx.x];
            sum_squares += s_data[i + threadIdx.x] * s_data[i + threadIdx.x];
        }
    }

    // Use blockReduce to compute total sum and sum_squares
    // But since we have a block of C threads, perhaps use a reduction loop
    // Alternatively, use atomicAdd, but that might be slow.

    // Alternatively, use a parallel reduction within the block.
    // This part is a bit tricky. Let's use a simple approach for now.

    // Wait, perhaps using a reduction across the block's threads:

    // First compute partial sums for each thread:
    // But with C threads, each can contribute their own value's contribution to the sum and sum_squares.

    // After __syncthreads(), each thread has their own value in s_data[c].

    // To compute sum of all elements in the block's C elements:
    // Each thread contributes their own value to the sum and sum_squares.

    // So sum = sum_{c=0}^{C-1} s_data[c]
    // sum_squares = sum_{c=0}^{C-1} s_data[c]^2

    // We can compute these with a parallel reduction.

    // Let's use a parallel reduction for sum and sum_squares:

    // Use a loop where each iteration halves the number of active threads:
    for (int stride = 1; stride < C; stride *= 2) {
        if (threadIdx.x % (2*stride) == 0) {
            int idx = threadIdx.x + stride;
            if (idx < C) {
                sum += s_data[idx];
                sum_squares += s_data[idx] * s_data[idx];
            }
        }
        __syncthreads();
    }

    // After reduction, thread 0 has the total sum and sum_squares
    float mean = 0.0f;
    float var = 0.0f;
    if (threadIdx.x == 0) {
        mean = sum / C;
        var = (sum_squares - (sum * sum)/C) / C;
    }

    // Broadcast mean and variance to all threads in the block
    __shared__ float shared_mean, shared_var;
    if (threadIdx.x == 0) {
        shared_mean = mean;
        shared_var = var;
    }
    __syncthreads();

    mean = shared_mean;
    var = shared_var;

    // Compute the normalized value for this channel
    float normalized = (val - mean) / sqrtf(var + eps);
    normalized = normalized * gamma[c] + beta[c];

    // Write to output
    int out_index = get_linear_index(b, c, d, h, w, C, D, H, W);
    output[out_index] = normalized;
}