#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ sum_weight_rows,
    scalar_t sum_biases,
    scalar_t* output,
    int batch_size,
    int in_features
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Compute sum_val = input • sum_weight_rows + sum_biases
    scalar_t sum_val = 0.0;
    for (int in = 0; in < in_features; ++in) {
        sum_val += input[batch_idx * in_features + in] * sum_weight_rows[in];
    }
    sum_val += sum_biases;

    // Apply subsequent operations as per original code
    // Step 2: Max over dim=1 (which is now a scalar)
    scalar_t max_val = sum_val;

    // Step 3: Mean over dim=1 (same as sum_val)
    scalar_t mean_val = max_val;

    // Step 4: First logsumexp
    scalar_t lse1 = mean_val;  // Because log(exp(x)) = x

    // Step 5: Second logsumexp
    scalar_t lse2 = lse1;      // Same as above

    // Store the final result
    output[batch_idx] = lse2;
}