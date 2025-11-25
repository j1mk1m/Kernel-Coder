class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_hswish_relu = conv_hswish_relu

    def forward(self, x):
        x = self.conv_hswish_relu.conv_hswish_relu_cuda(x, self.weight)
        return x