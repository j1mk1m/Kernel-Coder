model = Model(*get_init_inputs()).cuda()

inputs = get_inputs()
outputs = model(*inputs)

print(outputs.shape)