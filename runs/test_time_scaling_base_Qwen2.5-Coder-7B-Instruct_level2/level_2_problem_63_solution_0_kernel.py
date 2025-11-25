def gradcheck_custom_operators():
    # Create inputs and parameters
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    divisor = 2.0
    x = torch.randn(batch_size, in_features, requires_grad=True).cuda()

    # Initialize the original model and the custom model
    model_original = Model(in_features, out_features, divisor).cuda()
    model_new = ModelNew(in_features, out_features, divisor).cuda()

    # Forward pass through the original and custom models
    y_original = model_original(x)
    y_new = model_new(x)

    # Compute gradients
    grad_y_original = torch.randn_like(y_original).cuda()
    grad_y_new = torch.randn_like(y_new).cuda()

    # Backward pass through the original and custom models
    model_original.zero_grad()
    y_original.backward(grad_y_original)
    grad_x_original = x.grad

    model_new.zero_grad()
    y_new.backward(grad_y_new)
    grad_x_new = x.grad

    # Verify gradient consistency
    assert torch.allclose(grad_x_original, grad_x_new), "Gradient mismatch detected!"
    print("Gradient check passed!")