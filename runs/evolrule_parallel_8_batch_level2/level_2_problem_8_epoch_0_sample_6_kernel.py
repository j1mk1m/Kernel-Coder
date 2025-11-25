__global__ void sum_dim1_kernel(
    const float* input,
    float* output,
    int N, int C) {

    int n = blockIdx.x;

    __shared__ float sdata[16]; // C is 16

    if (threadIdx.x < C) {
        sdata[threadIdx.x] = input[n * C + threadIdx.x];
    }
    __syncthreads();

    for (int s = C/2; s > 0; s >>=1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[n] = sdata[0];
    }
}

torch::Tensor sum_dim1_cuda(torch::Tensor input) {
    int N = input.size(0);
    int C = input.size(1);

    auto output = torch::empty({N, 1, 1, 1, 1}, input.options());

    const dim3 threads(16);
    const dim3 blocks(N);

    sum_dim1_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C);

    return output;
}