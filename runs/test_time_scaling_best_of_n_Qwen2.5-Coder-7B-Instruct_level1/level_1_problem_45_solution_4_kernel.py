# Example usage
if __name__ == "__main__":
    model_new = ModelNew(kernel_size=11)
    inputs = get_inputs()
    outputs = model_new(inputs[0])
    print(outputs.shape)