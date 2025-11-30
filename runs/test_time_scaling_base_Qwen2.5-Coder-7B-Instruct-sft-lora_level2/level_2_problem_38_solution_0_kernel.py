inputs = get_inputs()
model = Model(*get_init_inputs())
outputs = model(inputs[0])
print(outputs.shape)