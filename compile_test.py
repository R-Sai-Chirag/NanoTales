import torch

# Force the compiler to be verbose so you can see if MSVC/Triton is failing
import torch._dynamo
torch._dynamo.config.verbose = True

def simple_math(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

# Compile the function
compiled_math = torch.compile(simple_math)

# Create dummy tensors on your GPU (or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(10, 10, device=device)
y = torch.randn(10, 10, device=device)

print("Starting compilation... (this will take a few seconds on the first run)")
result = compiled_math(x, y)
print("Success! Output shape:", result.shape)