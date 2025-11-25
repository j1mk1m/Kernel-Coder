model = Model(*get_init_inputs())
x = get_inputs()[0].cuda()
y = model(x).cpu().numpy()
print(y.shape)