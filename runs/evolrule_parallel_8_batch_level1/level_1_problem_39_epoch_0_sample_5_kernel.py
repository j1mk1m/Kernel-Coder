def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]