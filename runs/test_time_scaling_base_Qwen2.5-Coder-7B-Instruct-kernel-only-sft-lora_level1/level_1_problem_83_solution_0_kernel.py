class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.custom_function = custom_function

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.custom_function.custom_function_cuda(input)