class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose
        self.layer_norm = layer_norm
        self.gelu = gelu
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_cuda(x, self.weight, stride_d=self.stride[0], stride_h=self.stride[1], stride_w=self.stride[2], padding_d=self.padding[0], padding_h=self.padding[1], padding_w=self.padding[2])
        x = self.layer_norm.layer_norm_cuda(x, eps=self.eps)
        x = self.gelu.gelu_cuda(x)
        x = x * self.scaling_factor
        return x