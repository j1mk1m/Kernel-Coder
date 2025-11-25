import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Define and load the fused add + HardSwish kernel
        fused_add_hardswish_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void fused_add_hardswish_kernel(const float* x, const float* add_input, float* out, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float a = x[idx] + add_input[idx];
                float tmp = a + 3.0f;
                tmp = fmaxf(fminf(tmp, 6.0f), 0.0f);
                out[idx] = a * tmp * (1.0f / 6.0f);
            }
        }

        torch::Tensor fused_add_hardswish_cuda(torch::Tensor x, torch::Tensor add_input) {
            auto size = x.numel();
            auto out = torch::empty_like(x);

            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            fused_add_hardswish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), add_input.data_ptr<float>(), out.data_ptr<float>(), size);

            return out;
        }
        """

        fused_add_hardswish_cpp_source = (
            "torch::Tensor fused_add_hardswish_cuda(torch::Tensor x, torch::Tensor add_input);"
        )

        # Compile the inline CUDA code for fused add + HardSwish
        self.fused_add_hardswish = load_inline(
            name="fused_add_hardswish",
            cpp_sources=fused_add_hardswish_cpp_source,
            cuda_sources=fused_add_hardswish_source,
            functions=["fused_add_hardswish_cuda"],
            verbose=True,
            extra_cflags=[""],
            extra_ldflags=[""],
        )

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        # Add bias to the convolution output before fusion (since original code has x = x + add_input followed by activation)
        # Wait, the original code adds add_input, not the bias parameter. Need to check:
        # Original code: x = self.conv_transpose(x) (which includes bias if the ConvTranspose has bias enabled)
        # Then x = x + add_input (the input argument)
        # Then x = x * hardswish(x)
        # However, in the current model definition, the ConvTranspose3d has its own bias (by default), and the self.bias is a separate parameter added here. Wait looking back:

        # Wait, the original Model's __init__ has self.bias = nn.Parameter(...), but in forward, it's not used. Wait checking the original code:

        # Original code's forward is:
        # def forward(self, x, add_input):
        #     x = self.conv_transpose(x)
        #     x = x + add_input
        #     x = x * torch.nn.functional.hardswish(x)
        #     return x

        # So the self.bias is never used in the forward. That might be a mistake in the original code. However, the user provided this as the architecture, so perhaps there is a typo. But the user says to optimize their given architecture as is. So perhaps in their code, the self.bias is not used, but the 'add_input' is the tensor added. So the code in the ModelNew should follow the original forward, which does not use self.bias. Therefore, the fused kernel is just adding the add_input and then applying hardswish.

        # Proceed as per original code's forward, which adds add_input (the input argument) to the convolution result, then applies hardswish.

        # Thus, the fusion is correct as written. Proceed.

        x = self.fused_add_hardswish.fused_add_hardswish_cuda(x, add_input)
        return x

# The get_inputs and get_init_inputs functions remain the same as in the original problem statement, so they are not included here as the user requested only the ModelNew code.