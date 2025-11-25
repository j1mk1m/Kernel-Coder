def forward(self, x):
    x = self.matmul(x)
    x = torch.sigmoid(x) * x  # Swish activation
    x = x + self.bias
    x = self.group_norm(x)
    return x