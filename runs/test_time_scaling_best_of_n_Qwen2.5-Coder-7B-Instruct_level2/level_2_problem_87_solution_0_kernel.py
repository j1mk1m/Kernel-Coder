batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

model = ModelNew(in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2)

inputs = get_inputs()

output = model(inputs[0])

print(output.shape)