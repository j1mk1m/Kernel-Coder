class ModelNewFusedMatDiv(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.divisor = divisor
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

        # Fused matmul/div kernel
        fused_matdiv_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void fused_matmul_div(const float* x, const float* weight, const float* bias, float* out,
                                         int batch_size, int input_size, int output_size, float divisor) {
            int i = blockIdx.x;
            int j = blockIdx.y;

            if (i >= batch_size || j >= output_size)
                return;

            float sum = 0.0f;
            for (int k = 0; k < input_size; ++k) {
                sum += x[i * input_size + k] * weight[j * input_size + k];
            }
            sum += bias[j];
            sum /= divisor;

            out[i * output_size + j] = sum;
        }

        torch::Tensor fused_matmul_div_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                                            float divisor, int output_size) {
            const int batch_size = x.size(0);
            const int input_size = x.size(1);

            auto out = torch::empty({batch_size, output_size}, x.options());

            dim3 grid(batch_size, output_size);
            dim3 block(1);

            fused_matmul_div<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(),
                                             bias.data_ptr<float>(), out.data_ptr<float>(),
                                             batch_size, input_size, output_size, divisor);

            return out;
        }
        """
        fused_matdiv_cpp = "torch::Tensor fused_matmul_div_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float divisor, int output_size);"

        # GELU approximation kernel
        gelu_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void gelu_approx(const float* in, float* out, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = in[idx];
                float sigmoid_x = 1.0f / (1.0f + expf(-1.702f * x));
                out[idx] = x * sigmoid_x;
            }
        }

        torch::Tensor gelu_approx_cuda(torch::Tensor in) {
            int size = in.numel();
            auto out = torch::empty_like(in);

            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            gelu_approx<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), size);

            return out;
        }
        """
        gelu_cpp = "torch::Tensor gelu_approx_cuda(torch::Tensor in);"

        # Compile both kernels
        self.fused_matdiv = load_inline(
            name="fused_matdiv",
            cuda_sources=fused_matdiv_source,
            cpp_sources=fused_matdiv_cpp,
            functions=["fused_matmul_div_cuda"],
            verbose=True
        )

        self.gelu_op = load_inline(
            name="gelu_op",
            cuda_sources=gelu_source,
            cpp_sources=gelu_cpp,
            functions=["gelu_approx_cuda"],
            verbose=True
        )

    def forward(self, x):
        intermediate = self.fused_matdiv.fused_matmul_div_cuda(
            x, self.weight, self.bias, self.divisor, self.weight.size(0)
        )
        return self.gelu_op.gelu_approx_cuda(intermediate)

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, output_size, divisor]