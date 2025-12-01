class ModelNew(nn.Module):
       def __init__(self, num_classes=1000):
           super(ModelNew, self).__init__()
           self.conv1_weight = nn.Parameter(torch.randn(96, 3, 11, 11))
           self.conv1_bias = nn.Parameter(torch.randn(96))

       def forward(self, x):
           x = conv2d.conv2d_cuda(x, self.conv1_weight, pad_h=2, pad_w=2, stride_h=4, stride_w=4)
           return x