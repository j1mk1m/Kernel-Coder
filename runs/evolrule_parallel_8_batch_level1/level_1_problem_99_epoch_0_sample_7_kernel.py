import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class TripletMarginLossCUDA(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletMarginLossCUDA, self).__init__()
        self.margin = margin

        # Define CUDA kernel source code
        triplet_loss_source = f"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void triplet_loss_kernel(
            const float* __restrict__ anchor,
            const float* __restrict__ positive,
            const float* __restrict__ negative,
            float* output,
            int batch_size,
            int dim,
            float margin) 
        {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size) return;

            const float* a = anchor + idx * dim;
            const float* p = positive + idx * dim;
            const float* n = negative + idx * dim;

            float ap_dist_sq = 0.0f;
            float an_dist_sq = 0.0f;

            const float4* a4 = (const float4*)a;
            const float4* p4 = (const float4*)p;
            const float4* n4 = (const float4*)n;

            // Process 4 elements at a time using vector types
            for (int i = 0; i < dim/4; ++i) {{
                float4 ap_diff = a4[i] - p4[i];
                float4 an_diff = a4[i] - n4[i];
                ap_dist_sq += ap_diff.x*ap_diff.x + ap_diff.y*ap_diff.y + ap_diff.z*ap_diff.z + ap_diff.w*ap_diff.w;
                an_dist_sq += an_diff.x*an_diff.x + an_diff.y*an_diff.y + an_diff.z*an_diff.z + an_diff.w*an_diff.w;
            }}

            float ap_dist = sqrtf(ap_dist_sq);
            float an_dist = sqrtf(an_dist_sq);
            float loss_val = ap_dist - an_dist + margin;
            output[idx] = loss_val > 0.0f ? loss_val : 0.0f;
        }}

        torch::Tensor triplet_loss_cuda(
            torch::Tensor anchor,
            torch::Tensor positive,
            torch::Tensor negative,
            float margin) 
        {{
            const int batch_size = anchor.size(0);
            const int dim = anchor.size(1);

            auto output = torch::empty({{batch_size}}, anchor.options());

            const int block_size = 256;
            const int num_blocks = (batch_size + block_size - 1) / block_size;

            triplet_loss_kernel<<<num_blocks, block_size>>>(
                anchor.data_ptr<float>(),
                positive.data_ptr<float>(),
                negative.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                dim,
                margin
            );

            return output.mean();
        }}
        """

        # Define C++ declarations
        triplet_loss_cpp_source = (
            "torch::Tensor triplet_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);"
        )

        # Compile the CUDA code
        self.triplet_loss_cuda_mod = load_inline(
            name="triplet_loss_cuda",
            cpp_sources=triplet_loss_cpp_source,
            cuda_sources=triplet_loss_source,
            functions=["triplet_loss_cuda"],
            verbose=False
        )

    def forward(self, anchor, positive, negative):
        # Ensure tensors are contiguous for optimal memory access
        anchor = anchor.contiguous()
        positive = positive.contiguous()
        negative = negative.contiguous()

        # Compute loss using CUDA kernel
        return self.triplet_loss_cuda_mod.triplet_loss_cuda(
            anchor, positive, negative, self.margin
        )

class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.loss_fn = TripletMarginLossCUDA(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)