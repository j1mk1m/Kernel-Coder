class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu = gelu
    
    def forward(self, x):
        return self.gelu.gelu_cuda(x)