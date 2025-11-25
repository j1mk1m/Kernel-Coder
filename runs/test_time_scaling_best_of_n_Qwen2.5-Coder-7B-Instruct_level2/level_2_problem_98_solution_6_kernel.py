# Test the new model
batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

model_new = ModelNew(in_features, out_features, pool_kernel_size, scale_factor)
x = torch.rand(batch_size, in_features)
output = model_new(x)
print(output.shape)