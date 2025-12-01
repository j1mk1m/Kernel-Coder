model_new = ModelNew(kernel_size, stride, padding)
inputs = get_inputs()
output = model_new(inputs[0])
print(output.shape)  # Should print the shape after applying average pooling