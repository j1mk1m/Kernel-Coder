inputs = get_inputs()
model_new = ModelNew(*get_init_inputs())
outputs = model_new(inputs[0])
print(outputs.shape)