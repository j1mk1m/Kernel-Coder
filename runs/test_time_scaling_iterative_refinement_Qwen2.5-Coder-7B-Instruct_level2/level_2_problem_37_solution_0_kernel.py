class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.matmul_swish = matmul_swish

    def forward(self, x):
        x = self.matmul_swish.matmul_swish_cuda(x, self.weight)
        x = x + self.bias
        x = self.group_norm(x)
        return x