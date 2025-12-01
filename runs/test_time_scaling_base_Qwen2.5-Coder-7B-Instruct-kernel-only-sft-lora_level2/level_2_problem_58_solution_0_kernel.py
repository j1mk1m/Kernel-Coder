model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, bias_shape)
inputs = get_inputs()
output = model_new(inputs[0])
print(output.shape)