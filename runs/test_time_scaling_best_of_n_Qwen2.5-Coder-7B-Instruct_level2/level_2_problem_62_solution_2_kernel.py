batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512

model = ModelNew(input_size, hidden_size, num_groups)
inputs = get_inputs()
output = model(inputs[0])
print(output.shape)