model = ModelNew(in_channels=8, out_channels=64, kernel_size=3, 
                scaling_factor=2.0, bias_shape=(64,1,1), pool_kernel_size=4)
model.cuda()