if __name__ == "__main__":
    inputs = get_inputs()
    model_new = ModelNew(*get_init_inputs())
    outputs = model_new(inputs[0].cuda())
    print(outputs.shape)