import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.elementwise_mul = elementwise_mul

    def forward(self, x):
        x = self.conv(x)
        x = self.elementwise_mul.elementwise_mul_cuda(x, x)
        x = torch.tanh(x)
        return x

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

model_new = ModelNew(*get_init_inputs())
inputs = get_inputs()
output = model_new(inputs[0])
print(output.shape)