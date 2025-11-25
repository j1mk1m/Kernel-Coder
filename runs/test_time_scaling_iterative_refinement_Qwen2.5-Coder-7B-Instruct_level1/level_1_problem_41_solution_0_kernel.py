model_new = ModelNew(*get_init_inputs())
input_tensor = get_inputs()[0]
output_tensor = model_new(input_tensor)
print(output_tensor.shape)