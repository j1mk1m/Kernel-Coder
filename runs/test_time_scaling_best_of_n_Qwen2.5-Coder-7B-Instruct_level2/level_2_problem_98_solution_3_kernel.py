model_new = ModelNew(in_features, out_features, pool_kernel_size, scale_factor)
input_tensor = get_inputs()[0].cuda()
output_tensor = model_new(input_tensor)
print(output_tensor.shape)