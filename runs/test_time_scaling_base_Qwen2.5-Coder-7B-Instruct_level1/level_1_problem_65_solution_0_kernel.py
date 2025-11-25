model_new = ModelNew(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
input_tensor = get_inputs()[0]
output_tensor = model_new(input_tensor)
print(output_tensor.shape)  # Should print: torch.Size([8, 64, 512, 512])