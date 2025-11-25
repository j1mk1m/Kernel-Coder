model_new = ModelNew(dim)
inputs = get_inputs()
output = model_new(*inputs)
print(output.shape)  # Should match the expected output shape