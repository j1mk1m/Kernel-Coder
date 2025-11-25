model = MyModel().cuda()
input_tensor = torch.randn(32, 128).cuda()
output_tensor = model(input_tensor)
print(output_tensor.shape)