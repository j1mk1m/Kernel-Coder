# Ensure the code compiles and runs correctly
model_new = ModelNew(*get_init_inputs())
inputs = get_inputs()
output = model_new(inputs[0].cuda())
print(output.shape)