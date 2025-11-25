model = Model(input_size, hidden_size)
x = get_inputs()[0].cuda()
output = model(x.cuda())
print(output.shape)