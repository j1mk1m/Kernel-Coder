# Initialize weights for the convolutional transpose layer
in_channels = 32
out_channels = 64
kernel_size = 5
stride = 2
padding = 2

weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size).cuda()

# Create an instance of the new model
model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding)

# Get inputs
inputs = get_inputs()[0].cuda()

# Forward pass through the new model
output_new = model_new(inputs)

# Check if the outputs match between the old and new models
assert torch.allclose(output_ref, output_new), "The outputs do not match."