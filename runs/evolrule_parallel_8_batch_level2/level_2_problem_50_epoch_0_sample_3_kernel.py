class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 scale1, scale2, bias_shape):
        super().__init__()
        # Initialize convolution weight as a Parameter
        self.weight = nn.Parameter(
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding
            ).weight
        )
        # Keep scales as Parameters for training
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.kernel_size_pool = 2  # AvgPool kernel size
        self.stride = stride
        self.padding = padding
        self.fused_conv_transpose = fused_conv_transpose

    def forward(self, x):
        # Fused ConvTranspose3d + Scale1
        x = self.fused_conv_transpose.fused_conv_transpose_scale(
            x, self.weight, self.scale1.item(), self.stride, self.padding
        )
        
        # Remaining operations (to be replaced by fused kernels in full implementation):
        x = F.avg_pool3d(x, kernel_size=self.kernel_size_pool)
        x = x + self.bias
        x = x * self.scale2
        return x