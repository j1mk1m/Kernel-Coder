batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5

model_new = ModelNew(input_size, hidden_size, scaling_factor)
inputs = get_inputs()

output = model_new(inputs[0])
print(output.shape)  # Should print: torch.Size([1024, 8192])