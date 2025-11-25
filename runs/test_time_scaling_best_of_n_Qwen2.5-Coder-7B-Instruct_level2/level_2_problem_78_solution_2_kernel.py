model_new = ModelNew(*get_init_inputs())
inputs = get_inputs()
output = model_new(inputs[0])
print(output.shape)