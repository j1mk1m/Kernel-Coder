import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 1024  # Increased batch size
in_features = 8192  # Increased input features
out_features = 8192  # Increased output features

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        
        # Define and load the fused elementwise kernel
        fused_elementwise_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void fused_elementwise_kernel(const float* x, const float* y, float* out, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                out[idx] = (x[idx] + y[idx]) * y[idx];
            }
        }

        torch::Tensor fused_elementwise_cuda(torch::Tensor x, torch::Tensor y) {
            auto size = x.numel();
            auto out = torch::empty_like(x);
            
            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            fused_elementwise_kernel<<<num_blocks, block_size>>>(
                x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), size
            );
            
            return out;
        }
        """

        fused_elementwise_header = """
        torch::Tensor fused_elementwise_cuda(torch::Tensor x, torch::Tensor y);
        """

        self.fused_elementwise = load_inline(
            name="fused_elementwise",
            cpp_sources=fused_elementwise_header,
            cuda_sources=fused_elementwise_source,
            functions=["fused_elementwise_cuda"],
            verbose=True,
            extra_cflags=[""],
            extra_ldflags=[""],
        )

    def forward(self, x, y):
        x = self.bmm(x)
        x = self.instance_norm(x.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
        return self.fused_elementwise.fused_elementwise_cuda(x, y)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda(), torch.rand(batch_size, out_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]