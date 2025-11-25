batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1

model = ModelNew(in_features, out_features, max_dim)
input_tensor = get_inputs()[0].cuda()

output = model(input_tensor)
print(output.shape)  # Should print: torch.Size([1024, 8192])