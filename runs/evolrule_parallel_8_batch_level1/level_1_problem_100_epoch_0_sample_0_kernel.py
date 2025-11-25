import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the CUDA kernel for hinge loss computation
        hinge_loss_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* output, int batch_size, int dim) {
            extern __shared__ float shared[];
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch_size) {
                float loss = 1.0f - predictions[idx] * targets[idx];
                loss = (loss > 0.0f) ? loss : 0.0f;
                shared[threadIdx.x] = loss;
            }
            __syncthreads();

            // Thread block reduction
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    shared[threadIdx.x] += shared[threadIdx.x + s];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                atomicAdd(output, shared[0]);
            }
        }

        torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
            int batch_size = predictions.size(0);
            int dim_size = predictions.size(1);
            float* output = new float{0.0f};
            torch::Tensor loss_sum = torch::from_blob(output, {1}, predictions.options());

            const int block_size = 256;
            const int num_blocks = (batch_size + block_size - 1) / block_size;

            dim3 grid(num_blocks);
            dim3 block(block_size);
            int shared_size = block_size * sizeof(float);

            hinge_loss_kernel<<<grid, block, shared_size>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                output,
                batch_size,
                dim_size
            );

            float mean_loss = *output / batch_size;
            delete[] output;
            return torch::tensor(mean_loss, predictions.options());
        }
        """

        # Compile the CUDA kernel
        hinge_loss = load_inline(
            name="hinge_loss",
            cpp_sources="",
            cuda_sources=hinge_loss_source,
            functions=["hinge_loss_cuda"],
            verbose=True
        )

        self.hinge_loss_cuda = hinge_loss.hinge_loss_cuda

    def forward(self, predictions, targets):
        return self.hinge_loss_cuda(predictions, targets)

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape).cuda(), 
            (torch.randint(0, 2, (batch_size,)).float() * 2 - 1).cuda()]

def get_init_inputs():
    return []