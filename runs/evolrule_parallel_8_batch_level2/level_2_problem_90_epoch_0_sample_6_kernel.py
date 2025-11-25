model_original = Model(in_channels, out_channels, kernel_size, sum_tensor_shape)
   model_original.eval()  # Ensure no dropout/random effects
   inputs = get_inputs()
   with torch.no_grad():
       output_original = model_original(*inputs)