from torch.utils.cpp_extension import load_inline

custom_op_source = """
// CUDA source code here
"""

custom_op_cpp_source = """
// C++ interface here
"""

custom_op = load_inline(
    name="custom_op",
    cpp_sources=custom_op_cpp_source,
    cuda_sources=custom_op_source,
    functions=["your_custom_function"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)