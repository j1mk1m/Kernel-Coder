from torch.utils.cpp_extension import load_inline

custom_op_source = """
// Custom CUDA code here
"""

custom_op_cpp_source = """
// Custom C++ code here
"""

custom_op = load_inline(
    name="custom_op",
    cpp_sources=custom_op_cpp_source,
    cuda_sources=custom_op_source,
    functions=["custom_op_function"],
    verbose=True,
    extra_cflags=[],
    extra_ldflags=[],
)