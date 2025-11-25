class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = CustomConv2d(in_channels, out_channels, kernel_size)
        self.group_norm = CustomGroupNorm(groups, out_channels, eps=eps)
        self.tanh = CustomTanh()
        self.hard_swish = CustomHardswish()

    def forward(self, x):
        x_conv = self.conv(x)
        x_norm = self.group_norm(x_conv)
        x_tanh = self.tanh(x_norm)
        x_hard_swish = self.hard_swish(x_tanh)
        x_res = x_conv + x_hard_swish
        x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)
        return x_logsumexp