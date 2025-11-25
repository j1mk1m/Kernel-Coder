def test_correctness():
    model = Model(*get_init_inputs())
    model_new = ModelNew(*get_init_inputs())
    model_new.load_state_dict(model.state_dict())
    inputs = get_inputs()
    model.cuda()
    model_new.cuda()
    outputs = model(*inputs).cpu().detach()
    outputs_new = model_new(*inputs).cpu().detach()
    assert torch.allclose(outputs, outputs_new, rtol=1e-3, atol=1e-4), "The outputs are not close enough!"

def test_speed():
    # Run some speed tests here
    pass

test_correctness()
test_speed()