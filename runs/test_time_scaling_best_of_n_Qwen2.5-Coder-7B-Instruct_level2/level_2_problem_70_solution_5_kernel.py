model_original = Model(input_size, hidden_size, scaling_factor)
model_new = ModelNew()

x = get_inputs()[0]

output_original = model_original(x)
output_new = model_new(x)

assert torch.allclose(output_original, output_new), "The outputs do not match!"