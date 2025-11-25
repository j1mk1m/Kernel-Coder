model = Model(...)
model_new = ModelNew(...)
model_new.load_state_dict(model.state_dict())
model_new.eval()
with torch.no_grad():
    output1 = model(*inputs)
    output2 = model_new(*inputs)
    assert torch.allclose(output1, output2, atol=1e-3), "The outputs are not close!"