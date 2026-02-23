import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Current Device: {torch.cuda.current_device()}")

try:
    print("\nAttempting basic Tensor operations on GPU...")
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = torch.tensor([4.0, 5.0, 6.0]).cuda()
    z = x + y
    print(f"Result: {z}")
    print("Basic tensor addition successful!")
    
    print("\nAttempting torch.mean operation (where your code crashed)...")
    feats = torch.randn(10, 80).cuda()
    mean = torch.mean(feats, dim=1, keepdim=True)
    print("torch.mean successful!")

except Exception as e:
    print(f"\nFAILED with error: {e}")
    import traceback
    traceback.print_exc()
