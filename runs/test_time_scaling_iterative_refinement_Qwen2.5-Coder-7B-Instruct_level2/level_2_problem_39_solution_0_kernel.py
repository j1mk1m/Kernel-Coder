class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Replace nn.Linear with a custom CUDA kernel for matrix multiplication
        self.gemm = CustomGEMM(in_features, out_features)
        # Replace nn.Parameter with a custom CUDA kernel for scaling
        self.scale = CustomScale(scale_shape)
        # Replace nn.BatchNorm1d with a custom CUDA kernel for batch normalization
        self.bn = CustomBatchNorm(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scale
        x = self.bn(x)
        return x