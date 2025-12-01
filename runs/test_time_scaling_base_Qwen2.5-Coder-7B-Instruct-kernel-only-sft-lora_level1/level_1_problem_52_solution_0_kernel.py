import torch
import torch.nn as nn

# Test the ModelNew class
model = ModelNew(dim=1)
x = torch.rand(128, 4096, 4095)
output = model(x)
print(output.shape)  # Should print torch.Size([128, 4095])