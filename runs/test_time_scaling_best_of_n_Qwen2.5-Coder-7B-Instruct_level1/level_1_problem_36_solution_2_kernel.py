# Example usage of the optimized model
model_new = ModelNew(features)
input_data = get_inputs()[0]
output = model_new(input_data)
print(output.shape)