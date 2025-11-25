c++
// CUDA code for Layer Normalization
__global__ void layernorm_forward(const float* data, float* mean, float* var, float* normed_data, int n, int d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    float square_sum = 0.0f;

    for (int i = tid; i < n * d; i += blockDim.x * gridDim.x) {
        sum += data[i];
        square_sum += data[i] * data[i];
    }

    __shared__ float shared_mean[256];
    __shared__ float shared_var[256];

    if (tid < 256) {
        shared_mean[tid] = sum / (n * d);
        shared_var[tid] = square_sum / (n * d) - shared_mean[tid] * shared_mean[tid];
    }

    __syncthreads();

    sum = 0.0f;
    square_sum = 0.0f;

    for (int i = tid; i < 256; i++) {
        sum += shared_mean[i];
        square_sum += shared_var[i];
    }

    mean[0] = sum / 256;
    var[0] = square_sum / 256;

    __syncthreads();

    for (int i = tid; i < n * d; i += blockDim.x * gridDim.x) {
        normed_data[i] = (data[i] - mean[0]) / sqrt(var[0] + 1e-5);
    }
}

torch::Tensor layernorm_forward_cuda(torch::Tensor data) {
    auto n = data.size(0);
    auto d = data.size(1);
    auto mean = torch::zeros({1}, data.options());
    auto var = torch::zeros({1}, data.options());
    auto normed_data = torch::empty_like(data);

    const int block_size = 256;
    const int num_blocks = (n * d + block_size - 1) / block_size;

    layernorm_forward<<<num_blocks, block_size>>>(data.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), normed_data.data_ptr<float>(), n, d);

    return normed_data;
}