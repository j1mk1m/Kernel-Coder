class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.weight1 = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias1 = nn.Parameter(torch.empty(hidden_size))
        self.weight2 = nn.Parameter(torch.empty(output_size, hidden_size))
        self.bias2 = nn.Parameter(torch.empty(output_size))
        
        # Initialize weights and biases like PyTorch's Linear
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        bound1 = 1 / math.sqrt(input_size)
        nn.init.uniform_(self.bias1, -bound1, bound1)
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        bound2 = 1 / math.sqrt(hidden_size)
        nn.init.uniform_(self.bias2, -bound2, bound2)

        self.fused_linear_sigmoid = linear_sigmoid
        self.fused_linear_lse = linear_lse

    def forward(self, x):
        x = self.fused_linear_sigmoid.fused_linear_sigmoid_cuda(
            x, self.weight1, self.bias1)
        x = self.fused_linear_lse.fused_linear_lse_cuda(
            x, self.weight2, self.bias2)
        return x