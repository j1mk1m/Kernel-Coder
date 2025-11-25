# Load the inline CUDA code using torch.utils.cpp_extension.load_inline
my_custom_module = load_inline(
    name="my_custom_module",
    cpp_sources=my_custom_cpp_source,
    cuda_sources=my_custom_source,
    functions=["my_custom_function"],
    verbose=True,
    extra_cflags=[],
    extra_ldflags=[],
)

# Use the custom CUDA function in the new architecture
class ModelNew(nn.Module):
    def __init__(self, ...) -> None:
        super().__init__()
        self.my_custom_function = my_custom_module.my_custom_function

    def forward(self, ...):
        ...