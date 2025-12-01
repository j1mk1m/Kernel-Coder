model_new = ModelNew(in_channels, out_channels, kernel_size, sum_tensor_shape)
model_new.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))