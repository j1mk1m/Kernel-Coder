import timeit

model = Model(*get_init_inputs()).cuda()
inputs = get_inputs()[0].cuda()

start_time = timeit.default_timer()
output = model(inputs)
end_time = timeit.default_timer()

print(f"Original Model Time: {end_time - start_time} seconds")