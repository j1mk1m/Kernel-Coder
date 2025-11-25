model_new = ModelNew().cuda()
inputs = get_inputs()
output = model_new(inputs[0].cuda(), inputs[1].cuda())
print(output.shape)