class ModelNew(nn.Module):
    def __init__(self, ...):
        super(ModelNew, self).__init__()
        self.custom_conv = custom_conv

    def forward(self, ...):
        x = self.custom_conv.custom_convolution_cuda(x, weight, bias)
        ...
        return x