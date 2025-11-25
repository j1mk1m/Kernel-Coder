__global__ void compute_mean_var(
    const float* input,
    float* scaled,
    float* sum,
    float* sum_sq,
    int batch, int channels,
    int depth, int height, int width,
    float scale_factor) {

    int c = blockIdx.x;
    int tid = threadIdx.x;

    int total = batch * depth * height * width;
    __shared__ float shared_sum[256];
    __shared__ float shared_sum_sq[256];

    shared_sum[tid] = 0.0f;
    shared_sum_sq[tid] = 0.0f;

    for (int i = tid; i < total; i += blockDim.x) {
        int b = i / (depth * height * width);
        int rem = i % (depth * height * width);
        int d = rem / (height * width);
        rem %= (height * width);
        int h = rem / width;
        int w = rem % width;

        int in_idx = b * channels * depth * height * width +
                     c * depth * height * width +
                     d * height * width +
                     h * width +
                     w;

        float val = input[in_idx] * scale_factor;
        scaled[in_idx] = val;

        shared_sum[tid] += val;
        shared_sum_sq[tid] += val * val;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sum_sq[tid] += shared_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[c] = shared_sum[0];
        sum_sq[c] = shared_sum_sq[0];
    }
}

// Then apply batch norm
__global__ void apply_batch_norm(
    const float* scaled,
    float* out,
    const float* mean,
    const float* var,
    const float* gamma,
    const float* beta,
    int batch, int channels,
    int depth, int height, int width,
    float eps) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels * depth * height * width) return;

    int c = (idx / (depth * height * width)) % channels;
    int b = idx / (channels * depth * height * width);
    int rem = idx % (channels * depth * height * width);
    rem %= (depth * height * width);
    int d = rem / (height * width);
    rem %= (height * width);
    int h = rem / width;
    int w = rem % width;

    float x_scaled = scaled[idx];
    float m = mean[c];
    float v = var[c];
    float inv_std = 1.0f / sqrtf(v + eps);
    float gamma_val = gamma[c];
    float beta_val = beta[c];

    out[idx] = (x_scaled - m) * inv_std * gamma_val + beta_val;
}

// Define the functions in C++
torch::Tensor scaled_batch_norm_cuda(
    torch::Tensor input,
    float scale_factor,
    torch::Tensor batch_norm_gamma,
    torch::Tensor batch_norm_beta,
    torch::Tensor batch_norm_running_mean,
    torch::Tensor batch_norm_running_var,
    float eps) {

    auto batch = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    // Allocate tensors for intermediate results
    auto scaled = torch::empty_like(input);
    auto sum = torch::zeros({channels}, input.options());
    auto sum_sq = torch::zeros({channels}, input.options());

    dim3 block(256);
    dim3 grid(channels); // each block handles a channel

    compute_mean_var<<<grid, block>>>(
        input.data_ptr<float>(),
        scaled.data_ptr<float>(),
        sum.data_ptr<float>(),
        sum_sq.data_ptr<float>(),
        batch, channels, depth, height, width,
        scale_factor);

    // Compute mean and var on CPU (since these are per-channel and small)
    auto mean = sum / (batch * depth * height * width);
    auto var = (sum_sq - sum.pow(2) / (batch * depth * height * width)) / (batch * depth * height * width);

    // Prepare mean and var as tensors on GPU
    auto mean_gpu = mean.cuda();
    auto var_gpu = var.cuda();

    // Launch apply batch norm kernel
    auto total_elements = batch * channels * depth * height * width;
    dim3 grid2((total_elements + block.x - 1) / block.x);
    apply_batch_norm<<<grid2, block>>>(
        scaled.data_ptr<float>(),
        scaled.data_ptr<float>(), // output overwrites scaled
        mean_gpu.data_ptr<float>(),
        var_gpu.data_ptr<float>(),
        batch_norm_gamma.data_ptr<float>(),
        batch_norm_beta.data_ptr<float>(),
        batch, channels, depth, height, width,
        eps);

    return scaled;
}