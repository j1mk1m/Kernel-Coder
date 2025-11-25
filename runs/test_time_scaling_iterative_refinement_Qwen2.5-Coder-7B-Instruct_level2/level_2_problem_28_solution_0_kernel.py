from torch.utils.cpp_extension import load_inline

linear_module = load_inline(
    name="linear_module",
    cpp_sources="torch::Tensor linear_cuda(torch::Tensor A, torch::Tensor B);",
    cuda_sources=linear_source,
    functions=["linear_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

model_new = nn.Sequential(
    linear_module.linear_cuda,
    nn.BatchNorm2d(out_features),
    nn.ReLU(),
    nn.Conv2d(out_features, out_features, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(out_features, out_features),
    nn.Sigmoid()
)