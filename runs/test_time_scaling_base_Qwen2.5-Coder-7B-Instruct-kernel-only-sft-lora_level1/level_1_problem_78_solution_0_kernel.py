model = ModelNew(*get_init_inputs())
inputs = get_inputs()
output = model(inputs[0])
print(output.shape)