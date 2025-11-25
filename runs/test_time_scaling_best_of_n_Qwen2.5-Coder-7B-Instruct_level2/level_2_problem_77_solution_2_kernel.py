model_new = ModelNew(in_channels, out_channels, kernel_size, scale_factor)
input_tensor = get_inputs()[0]
output_tensor = model_new(input_tensor)
print(output_tensor.shape)