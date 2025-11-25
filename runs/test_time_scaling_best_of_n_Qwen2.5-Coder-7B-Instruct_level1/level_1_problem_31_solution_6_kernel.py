def main():
    model_new = ModelNew(alpha=get_init_inputs()[0])
    inputs = get_inputs()
    outputs = model_new(inputs[0])

if __name__ == "__main__":
    main()