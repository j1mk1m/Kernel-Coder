template <typename scalar_t>
__global__ void spatial_softmax_kernel(
    const scalar_t* input,
    scalar_t* output,
    int B,
    int C,
    int D,
    int H,
    int W) {
    // Each block handles a (B, C) channel
    int b = blockIdx.x;
    int c = blockIdx.y;
    
    int spatial_size = D * H * W;
    int tid = threadIdx.x;
    
    extern __shared__ scalar_t sdata[];
    scalar_t* shared_max = sdata;
    scalar_t* shared_sum = sdata + blockDim.x;
    
    // Initialize shared memory
    if (tid == 0) {
        shared_max[0] = -INFINITY;
        shared_sum[0] = 0.0;
    }
    __syncthreads();
    
    // Compute max
    for (int idx = tid; idx < spatial_size; idx += blockDim.x) {
        int d = idx / (H * W);
        int h = (idx % (H * W)) / W;
        int w = idx % W;
        int input_offset = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        scalar_t val = input[input_offset];
        if (val > shared_max[0]) {
            atomicMax(&shared_max[0], val);
        }
    }
    __syncthreads();
    
    // Compute exponentials and sum
    scalar_t max_val = shared_max[0];
    for (int idx = tid; idx < spatial_size; idx += blockDim.x) {
        int d = idx / (H * W);
        int h = (idx % (H * W)) / W;
        int w = idx % W;
        int input_offset = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        scalar_t exp_val = exp(input[input_offset] - max_val);
        atomicAdd(&shared_sum[0], exp_val);
    }
    __syncthreads();
    
    // Compute softmax and write
    scalar_t total_sum = shared_sum[0];
    for (int idx = tid; idx < spatial_size; idx += blockDim.x) {
        int d = idx / (H * W);
        int h = (idx % (H * W)) / W;
        int w = idx % W;
        int output_offset = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        int input_offset = output_offset; // input and output are separate
        scalar_t exp_val = exp(input[input_offset] - max_val);
        output[output_offset] = exp_val / total_sum;
    }
}

torch::Tensor spatial_softmax_cuda(
    torch::Tensor input,
    int B,
    int C,
    int D,
    int H,
    int W) {
    auto output = torch::empty_like(input);
    
    // Number of blocks: B*C
    dim3 blocks(B, C);
    dim3 threads(256); // Adjust based on spatial_size
    
    // Shared memory size: 2 * sizeof(scalar_t) per block?
    // Wait: shared_max and shared_sum are each a single scalar per block
    // So shared memory needed: 2 * sizeof(scalar_t)
    // But in the kernel code above, the shared memory is allocated as:
    // extern __shared__ scalar_t sdata[];
    // with sdata size blockDim.x * 2? Not sure, perhaps need to compute.
    // Each block needs space for shared_max (1 element) and shared_sum (1 element)
    // So total shared memory: 2 * sizeof(scalar_t)
    int shared_mem_size = 2 * sizeof(scalar_t);
    
    spatial_softmax_kernel<float><<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W);
    
    return output;
}