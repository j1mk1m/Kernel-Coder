class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.hardswish = hardswish
        self.mish = mish

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.conv.weight, self.conv.bias)
        x = subtraction.subtraction_cuda(x, self.subtract_value)
        x = self.hardswish.hardswish_cuda(x)
        x = self.pool(x)
        x = self.mish.mish_cuda(x)
        return x