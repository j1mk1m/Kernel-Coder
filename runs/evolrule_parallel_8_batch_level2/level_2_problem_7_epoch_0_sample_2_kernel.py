class ModelNew(nn.Module):
      def __init__(self, ...):
          super().__init__()
          self.conv = nn.Conv3d(...)
          self.fused_activations = fused_activations  # Loaded CUDA module

      def forward(self, x):
          x = self.conv(x)
          x, = self.fused_activations.fused_activations_cuda(x, 0.01)
          return x