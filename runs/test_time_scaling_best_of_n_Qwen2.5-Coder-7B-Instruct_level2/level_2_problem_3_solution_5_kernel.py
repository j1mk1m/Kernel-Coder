def check_model(original_model, new_model, inputs):
    original_output = original_model(*inputs)
    new_output = new_model(*inputs)
    assert torch.allclose(original_output, new_output, atol=1e-5), "Outputs do not match"