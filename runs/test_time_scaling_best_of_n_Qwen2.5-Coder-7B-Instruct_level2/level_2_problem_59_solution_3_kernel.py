model = ModelNew(in_features, out_features, scaling_factor)
inputs = get_inputs()

with torch.no_grad():
    output = model(inputs[0].cuda())
print(output.shape)