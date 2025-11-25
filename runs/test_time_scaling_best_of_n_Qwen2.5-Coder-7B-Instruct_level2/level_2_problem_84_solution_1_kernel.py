# Example usage
model_new = ModelNew(in_features, out_features, bn_eps, bn_momentum, scale_shape)
input_tensor = get_inputs()[0].cuda()
output_tensor = model_new(input_tensor)
print(output_tensor.shape)