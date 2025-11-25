batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

model = ModelNew(in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor)
inputs = get_inputs()

output = model(inputs[0])
print(output.shape)