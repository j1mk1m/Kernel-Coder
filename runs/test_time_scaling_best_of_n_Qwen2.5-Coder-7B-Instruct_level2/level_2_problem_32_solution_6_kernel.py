# Check if the model works correctly
assert output.shape == (batch_size, 1, height, width)