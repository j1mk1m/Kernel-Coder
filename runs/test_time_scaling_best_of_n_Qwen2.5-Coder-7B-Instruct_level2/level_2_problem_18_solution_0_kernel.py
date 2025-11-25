model_new = ModelNew(*get_init_inputs())
x = get_inputs()[0].cuda()
y = model_new(x)
print(y.shape)  # Should print torch.Size([1024, 1])