from torch.utils.cpp_extension import load_inline

source_code = """
// Your CUDA source code here
"""

cpp_source_code = """
// Your C++ source code here
"""

model_new = load_inline(
    name='my_model',
    cpp_sources=cpp_source_code,
    cuda_sources=source_code,
    functions=['my_function'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)