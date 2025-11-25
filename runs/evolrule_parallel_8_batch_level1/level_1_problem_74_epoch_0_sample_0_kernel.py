# Test code (not part of the required output)
if __name__ == "__main__":
    # Test the ModelNew against the original Model
    batch_size = 32
    in_channels = 32
    out_channels = 64
    kernel_size = 5
    length = 131072
    stride = 1
    padding = 0
    dilation = 3

    # Create original model
    original_model = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False).cuda()
    original_model.weight.data.copy_(ModelNew(in_channels, out_channels, kernel_size, stride, padding, dilation, False).weight.data)

    # Create new model
    new_model = ModelNew(in_channels, out_channels, kernel_size, stride, padding, dilation, False).cuda()

    # Generate input
    x = torch.randn(batch_size, in_channels, length, device='cuda')

    # Forward pass
    original_out = original_model(x)
    new_out = new_model(x)

    # Check if outputs are close
    print(torch.allclose(original_out, new_out, atol=1e-5))