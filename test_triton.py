import torch
import time

# Basic compile test
def simple_fn(x, y):
    return torch.sin(x) + torch.cos(y)

compiled_fn = torch.compile(simple_fn)

x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
y = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)

# First call compiles — will take 30-60 seconds, this is normal
print("Compiling... (first call is slow, this is normal)")
out = compiled_fn(x, y)
print("Compilation successful!")

# Second call uses compiled version
t0 = time.time()
for _ in range(100):
    out = compiled_fn(x, y)
torch.cuda.synchronize()
compiled_time = (time.time() - t0) / 100

eager_fn = simple_fn
t0 = time.time()
for _ in range(100):
    out = eager_fn(x, y)
torch.cuda.synchronize()
eager_time = (time.time() - t0) / 100

print(f"Eager:    {eager_time*1000:.2f}ms")
print(f"Compiled: {compiled_time*1000:.2f}ms")
print(f"Speedup:  {eager_time/compiled_time:.2f}x")