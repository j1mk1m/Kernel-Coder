import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # Custom fused kernel for linear + sum + max + mean + two logsumexp
        self.fused_kernel = load_inline(
            name="fused_operations",
            cpp_sources="""
            torch::Tensor fused_operations(torch::Tensor input);
            """,
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda_runtime.h>
            #include <vector>

            template <typename scalar_t>
            __global__ void fused_operations_kernel(
                scalar_t* __restrict__ input,
                scalar_t* __restrict__ output,
                int batch_size,
                int in_features,
                int out_features,
                const float* __restrict__ weight_data,
                const float* __restrict__ bias_data
            ) {{
                // Each thread handles one element in the batch
                int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (batch_idx >= batch_size) return;

                // Linear layer computation (matrix multiplication)
                float sum = 0.0;
                for (int i = 0; i < in_features; ++i) {{
                    float val = input[batch_idx * in_features + i];
                    for (int j = 0; j < out_features; ++j) {{
                        sum += val * weight_data[i * out_features + j];
                    }}
                }}
                sum += bias_data[batch_idx]; // Assuming bias is per batch? Wait, no, nn.Linear has a single bias per output feature.

                // Wait, correction: nn.Linear's bias is a 1D tensor of size out_features. So need to add bias[j] for each output feature j.
                // Wait the previous approach is wrong. Let's recompute correctly.

                // Correcting the matrix multiplication with bias
                // Each output feature is computed as input * weight_col + bias[j]
                // So for each batch element, output is of size out_features
                // But according to the original model's forward:
                // x = self.linear(x)  # (batch, out_features)
                // then sum over dim 1, etc.

                // Let's recompute properly:
                // Linear layer: out[i] = input * weight[i] + bias[i]
                // So for the fused kernel, need to compute the entire linear output first, then apply the reductions.

                // To optimize, since after linear, we sum over all features, then max (but sum is already a scalar per batch)
                // Wait the original model steps are:
                // After linear (batch, out_features)
                // sum over dim=1 (sum all features for each batch) to get (batch, 1)
                // then max over dim=1 (but this is a scalar per batch, so max of a single element is itself)
                // then mean over dim=1 (again scalar)
                // then two logsumexp over dim=1 (each time scalar)
                // Wait, the max, mean, and logsumexp operations after the sum are redundant?

                // Wait let's re-examine the original forward:
                x = self.linear(x)  # (batch, out_features)
                x = torch.sum(x, dim=1, keepdim=True) # (batch, 1)
                x = torch.max(x, dim=1, keepdim=True)[0] # (batch, 1)
                x = torch.mean(x, dim=1, keepdim=True) # (batch, 1)
                x = torch.logsumexp(x, dim=1, keepdim=True) # (batch, 1)
                x = torch.logsumexp(x, dim=1, keepdim=True) # (batch, 1)
                return x

                // Let's see: after sum, each x is a vector of size (batch, 1). Then taking max over dim 1 (the only element) gives same as the original value. Similarly for mean. So those operations are redundant and can be removed. However, the problem states to preserve the same operations unless fused. So we need to ensure that all steps are applied as per the original model, even if they are redundant.

                // Therefore, the fused kernel must perform all steps as per original, even if some steps are redundant.

                // Compute linear layer output first:
                // For each batch element:
                // Compute sum over input * weight (matrix multiplication)
                // Add the bias (per output feature)

                // However, the subsequent steps (sum over features) reduces the output to a scalar per batch element.
                // Thus, maybe we can optimize the computation by fusing the linear layer with the sum.

                // Let me re-calculate:

                // The linear layer's output is x_linear = input @ weight.T + bias
                // Then, sum over all features (dim=1) gives sum_x = sum(x_linear[i] for i in 0..out_features-1)
                // Therefore, the total sum is equal to (input @ (sum over weight columns) ) + sum(bias)

                // Therefore, we can precompute the sum of the weight columns and the sum of the bias, then compute a single dot product.

                // This is a significant optimization opportunity!

                // Let's compute the weight_sum as sum over each column of weight (i.e., for each output feature j, sum over all input features i of weight[i][j])

                // So weight_sum is a vector of length out_features where weight_sum[j] = sum_i weight[i][j]

                // Then, the sum over all features of linear output is:
                // sum_j [ (input * weight[:,j] ) + bias[j] ] = sum_j (input @ weight[:,j]) + sum(bias)
                // Wait no: sum over j of (input * weight[:,j] + bias[j]) = input @ (sum(weight columns)) + sum(bias)

                // Therefore, the total sum can be computed as (input_vector) • (weight_sum_vector) + total_bias_sum

                // This is much more efficient than computing the full linear layer and then summing.

                // So in the fused kernel, we can precompute weight_sum and total_bias_sum once.

                // So first, precompute these values outside the kernel.

                // Let's proceed with this optimized approach.

                // Let's first compute the total_sum = input_vector • weight_sum + total_bias_sum

                // Then, apply max (which is same as the value since it's a single element)
                // Then mean (same value)
                // Then logsumexp twice.

                // Let's compute all these steps in the kernel.

                // First, precompute the weight_sum and total_bias_sum:

                // weight is a matrix of shape (in_features, out_features)
                // weight_sum is a vector of length out_features, sum over rows (input features)
                // total_bias_sum is the sum of all elements in the bias vector (size out_features)

                // So first, outside the kernel, compute these:

                // Assuming weight and bias are passed as parameters.

                // The fused kernel needs to take the input tensor, weight, and bias as inputs.

                // Wait, but in the original model, the linear layer's parameters are stored in self.linear.weight and self.linear.bias.

                // However, in the fused kernel, we need to include these parameters. But since we are replacing the entire sequence of operations with a custom kernel, perhaps we can pass the parameters as additional inputs to the kernel function.

                // However, in PyTorch's inline CUDA extensions, the kernel function can take tensors as inputs. So, the fused_operations function can accept input, weight, bias tensors.

                // Therefore, in the fused_operations_cuda function, we can have:

                // torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {

                // Then, inside the kernel, we can access weight and bias data.

                // So, in the kernel code:

                // Precompute weight_sum and total_bias_sum as temporary variables.

                // Let's compute weight_sum and total_bias_sum:

                // Compute weight_sum:
                // For each j in 0..out_features-1:
                // weight_sum[j] = sum_{i=0 to in_features-1} weight[i][j]

                // Since weight is stored in row-major order (or column-major?), assuming it's a tensor of shape (in_features, out_features), so the first dimension is in_features.

                // So the weight data is stored as weight[0][0], weight[0][1], ..., weight[0][out_features-1], weight[1][0], etc.

                // To compute the sum over the rows (input features) for each column (output feature):

                // We can precompute this outside the kernel, but since the kernel is called per input, perhaps we can compute it once and pass it as a tensor.

                // Alternatively, compute it inside the kernel, but that would require a reduction over the weight tensor for each kernel call, which is inefficient.

                // To optimize, the weight_sum and total_bias_sum can be precomputed once when initializing the fused kernel, but since the fused_operations function is called every time, we need to have them as input tensors.

                // Therefore, the fused_operations_cuda function will need to compute these sums each time? No, that would be inefficient.

                // Alternatively, perhaps we can structure the code such that the fused_operations kernel is initialized with the weight and bias, but in PyTorch's inline extension, the CUDA function is called each time, so we need to pass the weight and bias as tensors each time.

                // To make this efficient, let's compute the weight_sum and total_bias_sum as tensors once when initializing the fused_operations module.

                // Hmm, this is getting complicated. Let's proceed step by step.

                // Let's first assume that the fused kernel will receive the weight and bias tensors as inputs, and compute the necessary sums on the fly.

                // So in the kernel:

                // Compute for each batch element:

                // 1. Compute the sum over all features of the linear output.

                // To compute this sum efficiently:

                // The linear output's sum over features is equal to (input • weight_sum) + total_bias.

                // So let's compute weight_sum and total_bias first.

                // weight_sum is a vector of length out_features, sum over rows (input features).

                // total_bias is the sum over all elements of the bias vector.

                // So in the kernel, for each batch element:

                float total_sum = 0.0;

                // Compute weight_sum and total_bias:
                // Since weight and bias are tensors passed to the kernel, we need to loop over them.

                // However, this would require each thread to loop over all elements of weight and bias, which is not efficient.

                // Therefore, better to precompute these outside the kernel.

                // So the plan is:

                // Precompute weight_sum and total_bias once when initializing the fused kernel.

                // But how to do that in the inline CUDA code?

                // Alternatively, we can compute them in the fused_operations_cuda function before launching the kernel.

                // Here's the plan:

                // In the fused_operations_cuda function:

                // 1. Compute weight_sum as sum over rows of weight (size out_features)
                // 2. Compute total_bias = sum(bias)

                // Then, in the kernel, each thread can use these precomputed values.

                // Therefore, in the kernel function:

                // The kernel will have access to weight_sum and total_bias as parameters.

                // So, modifying the kernel parameters:

                // The kernel function signature in the CUDA code:

                // __global__ void fused_operations_kernel(
                //     scalar_t* input,
                //     scalar_t* output,
                //     int batch_size,
                //     int in_features,
                //     int out_features,
                //     const float* weight_sum_data,
                //     float total_bias
                // )

                // Then, in the fused_operations_cuda function:

                auto weight = ...; // passed as input to fused_operations
                auto bias = ...;

                // Compute weight_sum:
                auto weight_sum = torch::sum(weight, /*dim=*/0);
                // Compute total_bias = bias.sum()

                float total_bias = bias.sum().item<float>();

                // Then, pass weight_sum_data and total_bias to the kernel.

                // Now, in the kernel:

                // For each batch element (thread):

                // Compute input's dot product with weight_sum:

                float input_dot = 0.0;
                for (int i = 0; i < in_features; ++i) {{
                    input_dot += input[batch_idx * in_features + i];
                }}
                input_dot *= weight_sum_val; // Wait, no, weight_sum is a vector of length out_features.

                // Wait no: the input is a vector of length in_features, and weight_sum is a vector of length out_features where each element is the sum over input features for that output feature.

                // The dot product between input and weight_sum is sum_{i} input[i] * (sum_j weight[i][j])

                // Wait, no. Wait the input is of size in_features, and weight_sum is of size out_features, each element weight_sum[j] = sum_i weight[i][j].

                // The sum over all output features of (input • weight[:,j] + bias[j]) equals input • (sum_j weight[:,j]) + sum_j bias[j]

                // So, the total_sum = (input • weight_sum_vector) + total_bias

                // So the input's contribution is a scalar: (sum_i input[i]) * (sum_j weight_sum[j])

                // Wait no:

                // Let me clarify:

                The total_sum = sum_{j=0 to out_features-1} [ (sum_{i=0 to in_features-1} input[i] * weight[i][j]) + bias[j] ]

                = sum_{j} [ (input • weight_col_j) + bias[j] ]

                = input • (sum_j weight_col_j) ) + sum_j bias[j]

                where weight_col_j is the j-th column of weight (size in_features).

                The sum over columns of the weight columns is the vector where each element (for input feature i) is sum_j weight[i][j].

                However, the sum over all columns (output features) of each weight_col_j's elements gives for each input feature i: sum_j weight[i][j].

                Wait, no: sum_j weight_col_j (each column summed?) no, the sum over columns would be different.

                Let me re-express:

                The term (input • weight_col_j) for each j is the dot product between the input vector and the j-th column of the weight matrix.

                Summing over all j gives the sum over all such dot products plus the bias sum.

                To compute this efficiently, the total_sum can be rewritten as:

                total_sum = input • (sum_{j} weight_col_j) + sum_{j} bias[j]

                where sum_{j} weight_col_j is a vector of length in_features, where each element i is the sum over j of weight[i][j].

                Let me denote this vector as weight_row_sum.

                Therefore, the total_sum is:

                (input • weight_row_sum) + total_bias

                So the weight_row_sum is a vector of length in_features, where each element i is sum_{j} weight[i][j].

                Therefore, the kernel can compute the total_sum as the dot product between input and weight_row_sum, plus total_bias.

                This is a crucial optimization, as it reduces the computation from O(in_features * out_features) to O(in_features).

                Given that in_features and out_features are both 8192, this reduces the computation from ~67 million operations to ~8k operations per batch element.

                This is a massive optimization.

                So, to implement this, the kernel needs to:

                1. Precompute weight_row_sum and total_bias.

                2. Compute input's dot product with weight_row_sum.

                3. Add total_bias.

                4. Apply the subsequent operations (max, mean, logsumexp twice).

                Now, proceeding with this plan.

                First, compute weight_row_sum as the sum over columns of the weight matrix.

                weight_row_sum[i] = sum_{j} weight[i][j]

                total_bias = sum(bias)

                Now, in the kernel:

                For each batch element:

                total_sum = (sum_{i} input[i] * weight_row_sum[i]) + total_bias

                Then:

                x = total_sum  # after sum (dim=1) gives a scalar per batch element

                x = max(x) → same as x since it's a scalar

                x = mean(x) → same as x

                Then apply logsumexp once on x (dim=1), then again.

                Wait, logsumexp of a scalar is log(exp(x)), which is x. Wait no:

                logsumexp of a single element x over dim=1 (which has size 1) is log(exp(x)) = x.

                So applying logsumexp twice on a scalar would give x each time.

                So the entire sequence after the linear layer and sum reduces to x = x (no change).

                Wait, this means that all the operations after the linear layer's sum are redundant and can be removed.

                But the problem states that the ModelNew must have the same operations as the original, unless fused.

                However, the original model's steps after the linear layer are:

                x = torch.sum(...) → scalar per batch
                x = torch.max(x, dim=1, ...) → same scalar
                x = torch.mean(...) → same scalar
                x = logsumexp(...) → same scalar
                x = logsumexp(...) → same scalar

                So all those operations after the sum are redundant and do not change the output.

                However, the problem requires that the new model produces the same outputs as the original. Since all those operations after the sum are redundant (they do not change the value), the output is the same as the sum followed by the two logsumexp operations which also do not change the value. So the final output is the same as the initial sum.

                But the problem states to preserve the same operations unless fused. Since they are redundant, but we can choose to replace them with a single kernel.

                Therefore, the fused kernel can compute the initial sum (via the optimized method), then apply the subsequent operations (even if redundant) in the kernel.

                However, since those operations do nothing, we can skip them in the kernel, but the problem requires the same operations, so we need to ensure that all steps are applied, even if they do nothing.

                For correctness, even if they are redundant, we must simulate their effect.

                But in code, since applying max, mean, and logsumexp twice on a scalar will leave it unchanged, the kernel can just output the initial sum (plus the bias) after applying all the steps, but the net result is the same.

                Therefore, in the kernel:

                After computing total_sum = (input • weight_row_sum) + total_bias,

                then:

                // Apply max (dim=1, which is a scalar → no change)
                x = total_sum

                // Apply mean (dim=1 → same value)
                x = x

                // Apply first logsumexp: log(exp(x)) = x
                x = x

                // Apply second logsumexp: same as before
                x = x

                So the final output is the same as total_sum.

                Therefore, the entire computation reduces to total_sum.

                This is a crucial insight. However, the problem requires that we replace the operators with custom CUDA kernels, even if they are redundant. Therefore, we can choose to implement the fused kernel to compute the total_sum, then apply all the subsequent operations in the kernel (even though they are no-ops), to ensure that the operations are present and the code is correct.

                Alternatively, if we can prove that the subsequent operations do not change the result, we can omit them, but the problem requires that the same operations are applied (unless fused into the kernel). Since we are fusing all the operations into one kernel, it's acceptable.

                Therefore, the fused kernel can compute total_sum and return it as the final output, since all subsequent operations have no effect.

                Thus, the kernel's final output is simply the total_sum.

                Therefore, the fused kernel can be simplified to compute the linear layer's sum efficiently, then return that as the final result.

                Therefore, the fused kernel's code can be written as follows:

                // Compute weight_row_sum and total_bias in the host code.

                // In the kernel:

                // For each batch element:

                float total_sum = 0.0;

                // Compute the dot product between input and weight_row_sum
                for (int i = 0; i < in_features; ++i) {
                    total_sum += input[batch_idx * in_features + i] * weight_row_sum[i];
                }
                total_sum += total_bias;

                // The subsequent operations (max, mean, logsumexp twice) do not change the value, so we can skip them.

                output[batch_idx] = total_sum;

                // But to strictly follow the problem's requirement to apply all operations, we can do:

                // Apply max (no change)
                float current_val = total_sum;

                // Apply mean (no change)
                current_val = current_val;

                // Apply first logsumexp: log(exp(current_val)) = current_val
                current_val = current_val;

                // Apply second logsumexp: same as before
                current_val = current_val;

                output[batch_idx] = current_val;

                // But this is redundant and can be optimized away.

                // Therefore, the final output is total_sum.

                Now, let's write the kernel code accordingly.

                First, precompute weight_row_sum and total_bias in the fused_operations_cuda function.

                Here's the full CUDA code for the fused kernel:

                // CUDA kernel definition
                template <typename scalar_t>
                __global__ void fused_operations_kernel(
                    scalar_t* __restrict__ input_data,
                    scalar_t* __restrict__ output_data,
                    const int batch_size,
                    const int in_features,
                    const int out_features,
                    const float* __restrict__ weight_row_sum_data,
                    const float total_bias
                ) {
                    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (batch_idx >= batch_size) return;

                    float total_sum = 0.0f;

                    // Compute the dot product between input and weight_row_sum
                    for (int i = 0; i < in_features; ++i) {
                        total_sum += input_data[batch_idx * in_features + i] * weight_row_sum_data[i];
                    }

                    total_sum += total_bias;

                    // Apply subsequent operations (which are no-ops)
                    // Max (dim=1) → same value
                    // Mean (dim=1) → same value
                    // logsumexp (dim=1) twice → same value

                    output_data[batch_idx] = total_sum;
                }

                // Host function to launch the kernel
                torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
                    const int batch_size = input.size(0);
                    const int in_features = input.size(1);
                    const int out_features = weight.size(1);

                    // Compute weight_row_sum: sum over columns of weight (each row's elements summed)
                    auto weight_row_sum = torch::sum(weight, 1); // dim=1 sums over columns (since weight is [in, out])
                    const float* weight_row_sum_data = weight_row_sum.data_ptr<float>();

                    // Compute total_bias: sum of all elements in bias
                    float total_bias = bias.sum().item<float>();

                    // Output tensor of shape (batch_size, 1)
                    auto output = torch::empty({batch_size, 1}, input.options());

                    const int threads_per_block = 256;
                    const int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;

                    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_operations_cuda", ([&] {
                        fused_operations_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                            input.data_ptr<scalar_t>(),
                            output.data_ptr<scalar_t>(),
                            batch_size,
                            in_features,
                            out_features,
                            weight_row_sum_data,
                            total_bias
                        );
                    }));

                    return output;
                }

                """,
            extra_cuda_cflags=['-std=c++14'],
            functions=['fused_operations_cuda'],
            verbose=True
        )

        # Save the parameters from the original model's linear layer
        # Wait, but in the original model, the parameters are part of the Linear module.
        # However, in the new model, since we're using a custom kernel, we need to pass the weight and bias tensors explicitly each time.

        # To handle this, we can store the original linear layer's parameters and pass them to the kernel.

        # So we keep the original linear layer to get its parameters.
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Extract weight and bias from the linear layer
        weight = self.linear.weight
        bias = self.linear.bias

        # Call the fused kernel with input x, weight, bias
        # The fused_operations_cuda function is part of the loaded module
        # The kernel requires input, weight, bias as tensors
        # The output will be (batch_size, 1)
        output = self.fused_kernel.fused_operations_cuda(x, weight, bias)

        return output

def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    in_features = 8192
    out_features = 8192
    return [in_features, out_features]