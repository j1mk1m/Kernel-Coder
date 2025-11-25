custom_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_conv_kernel(...) {
    ...
}

torch::Tensor custom_conv_cuda(torch::Tensor ...) {
    ...
}
"""

custom_conv_cpp_source = "torch::Tensor custom_conv_cuda(torch::Tensor ...);"

custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=custom_conv_cpp_source,
    cuda_sources=custom_conv_source,
    functions=["custom_conv_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)