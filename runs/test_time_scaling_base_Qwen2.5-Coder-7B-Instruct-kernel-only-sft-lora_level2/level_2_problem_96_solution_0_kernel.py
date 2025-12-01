from torch.utils.cpp_extension import load_inline

model_new_source = """
// Add your custom CUDA kernels here
"""

model_new_cpp_source = (
    "void my_custom_function();"
)

model_new = load_inline(
    name="model_new",
    cpp_sources=model_new_cpp_source,
    cuda_sources=model_new_source,
    functions=["my_custom_function"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)