class ModelNew(nn.Module):
                def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
                    super(ModelNew, self).__init__()
                    self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                    self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

                def forward(self, x):
                    x = self.conv_transpose(x)
                    x = logsumexp.logsumexp_cuda(x)
                    x = post_logsumexp.post_logsumexp_cuda(x, self.bias.item())
                    return x