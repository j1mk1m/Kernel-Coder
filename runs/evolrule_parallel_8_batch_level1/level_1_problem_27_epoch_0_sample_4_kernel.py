# Test correctness
model_original = Model()
model_new = ModelNew()

x = torch.randn(batch_size, dim).cuda()
original_out = model_original(x)
new_out = model_new(x)

assert torch.allclose(original_out, new_out, atol=1e-5), "Outputs do not match"

# Test performance
import time

def measure_time(func, x, num_runs=1000):
    x_gpu = x.cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        func(x_gpu)
        torch.cuda.synchronize()
    return (time.time() - start) / num_runs

original_time = measure_time(model_original, x)
new_time = measure_time(model_new, x)

print(f"Original: {original_time:.6f} s/iter")
print(f"Optimized: {new_time:.6f} s/iter")