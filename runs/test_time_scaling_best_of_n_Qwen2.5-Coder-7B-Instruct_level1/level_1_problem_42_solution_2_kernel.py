# Check if the outputs are the same
output_original = model(inputs[0])
output_new = model_new(inputs[0])
assert torch.allclose(output_original, output_new)