class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.seLU = selu
    
    def forward(self, x):
        return self.seLU.seLU_forward_cuda(x)