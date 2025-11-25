# Example usage:
model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape)
x, add_input = get_inputs()
output = model_new(x, add_input)
print(output.shape)  # Should print torch.Size([128, 64, 32, 32, 32])