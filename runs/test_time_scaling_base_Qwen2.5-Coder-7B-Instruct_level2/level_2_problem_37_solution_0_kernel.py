import timeit

model = Model(*get_init_inputs()).to("cuda")
inputs = get_inputs()[0].to("cuda")

original_time = timeit.timeit(lambda: model(inputs), number=100)
print(f"Original time: {original_time} seconds")

model_new = ModelNew(*get_init_inputs()).to("cuda")
inputs = get_inputs()[0].to("cuda")

new_time = timeit.timeit(lambda: model_new(inputs), number=100)
print(f"New time: {new_time} seconds")