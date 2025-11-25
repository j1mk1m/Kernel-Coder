model_new = ModelNew(*get_init_inputs())
x = get_inputs()[0].cuda()
output = model_new(x)
print(output.shape)