from torch.utils.cpp_extension import load_inline

my_custom_module = load_inline(
    name="my_custom_module",
    cpp_sources=my_custom_cpp_source,
    cuda_sources=my_custom_source,
    functions=["my_custom_function"],
    verbose=True,
    extra_cflags=[],
    extra_ldflags=[],
)