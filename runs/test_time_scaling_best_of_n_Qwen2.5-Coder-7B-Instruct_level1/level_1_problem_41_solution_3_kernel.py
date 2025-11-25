import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

...

class ModelNew(nn.Module):
    ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...