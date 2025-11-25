torch::Tensor fused_gn_leakyrelu_scale_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int G,
    float eps,
    float negative_slope,
    float scale_factor) {

    int B = x.size(0);
    int H = x.size(1);
    int group_size = H / G;

    // Check dimensions
    assert(gamma.size(0) == H);
    assert(beta.size(0) == H);

    auto out = torch::empty_like(x);

    const int threads_per_block = 256;
    const int blocks = G; // One block per group

    // Calculate shared memory needed: 2 (for shared_sums) + 2 * threads_per_block
    int shared_mem = 2 * sizeof(float) + 2 * threads_per_block * sizeof(float);

    fused_gn_leakyrelu_scale_kernel<float>
        <<<blocks, threads_per_block, shared_mem>>>(
            x.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            out.data_ptr<float>(),
            B,
            H,
            G,
            group_size,
            eps,
            negative_slope,
            scale_factor);

    return out;
}