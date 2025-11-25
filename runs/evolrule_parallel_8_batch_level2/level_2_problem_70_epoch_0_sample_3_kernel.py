import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
torch::Tensor fused_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor, int batch_size, int input_size, int hidden_size);
"""

cuda_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float scaling_factor,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * hidden_size) return;

    int batch = tid / hidden_size;
    int hidden = tid % hidden_size;

    float gemm_val = bias[hidden];
    for (int i = 0; i < input_size; ++i) {
        int weight_idx = i * hidden_size + hidden;
        gemm_val += input[batch * input_size + i] * weight[weight_idx];
    }

    float sig = 1.0f / (1.0f + expf(-gemm_val));
    float scaled = sig * scaling_factor;
    float result = scaled + gemm_val;

    output[batch * hidden_size + hidden] = result;
}

torch::Tensor fused_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor, int batch_size, int input_size, int hidden_size) {
    auto output = torch::empty({batch_size, hidden_size}, input.options());
    
    int threads_per_block = 256;
    int blocks_per_grid = (batch_size * hidden_size + threads_per_block - 1) / threads_per_block;
    
    fused_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );
    
    return output;
}
"""

fused_cuda_module = load_inline(
    name="fused_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.register_buffer('scaling_factor', torch.tensor(scaling_factor, dtype=torch.float32))

    def forward(self, x):
        batch_size = x.size(0)
        input_size = x.size(1)
        hidden_size = self.gemm.out_features

        weight = self.gemm.weight.contiguous()
        bias = self.gemm.bias.contiguous()
        scaling_factor = self.scaling_factor.item()

        output = fused_cuda_module.fused_cuda(
            x.contiguous(),
            weight,
            bias,
            scaling_factor,
            batch_size,
            input_size,
            hidden_size
        )
        return output

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]

# The batch_size, input_size, hidden_size, scaling_factor are defined as per the original code's variables.
# But in the problem's original code, they are global variables. So in the final code, we need to include them.
# Wait, in the original code given, the variables are defined outside the Model class.

# Wait, the user's original code has:

# batch_size = 1024
# input_size = 8192
# hidden_size = 8192
# scaling_factor = 2.0

# def get_inputs():
#     return [torch.rand(batch_size, input_size)]

# def get_init_inputs():
#     return [input_size, hidden_size, scaling_factor]

# So in the new code, these variables should be present, but in the problem's instructions, when we output the code, do we need to include them?

The problem says to output the new architecture (ModelNew) and the get_inputs and get_init_inputs. The variables like batch_size, etc., are required in the original code's get_init_inputs and get_inputs. So in the new code, the user must define them or include them in the get_init_inputs.

Wait, the problem states to output the code for ModelNew and the get_inputs and get_init_inputs functions. The original code had those variables as global variables. Therefore, in the new code, they should be included as well.

So adding those global variables at the top:

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0

But in the problem's example, the get_init_inputs returns the parameters needed to initialize the model, which for the original Model was [input_size, hidden_size, scaling_factor]. So the ModelNew's __init__ also takes those parameters, so the get_init_inputs should return them.

Therefore, the complete code would be: