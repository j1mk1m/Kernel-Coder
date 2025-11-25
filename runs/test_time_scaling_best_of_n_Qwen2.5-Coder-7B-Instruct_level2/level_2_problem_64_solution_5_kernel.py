def get_inputs():
    # randomly generate input tensors based on the model architecture
    batch_size = 1024
    in_features = 8192
    x = torch.randn(batch_size, in_features).cuda()
    return [x]