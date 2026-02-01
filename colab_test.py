"""
Google Colab CUDA Test Script
Upload this along with the homework folder and bignet.pth
"""
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Test 3-bit quantization on CUDA
from homework.lower_precision import BigNet3Bit, Linear3Bit, block_quantize_3bit, block_dequantize_3bit
from homework.bignet import BigNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Test 1: Simple quantization round-trip
print("\n=== Test 1: Quantization Round-Trip ===")
test_weight = torch.randn(1024).to(device)
packed, norm = block_quantize_3bit(test_weight.cpu(), group_size=32)
reconstructed = block_dequantize_3bit(packed, norm, group_size=32)
reconstructed = reconstructed.to(device)
diff = (test_weight - reconstructed).abs()
print(f"Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")

# Test 2: Linear3Bit layer
print("\n=== Test 2: Linear3Bit Layer ===")
layer = Linear3Bit(128, 128, bias=True, group_size=32)
weight = torch.randn(128, 128)
bias = torch.randn(128)
layer.load_state_dict({'weight': weight, 'bias': bias})

# Move to CUDA
layer = layer.to(device)
x = torch.randn(10, 128).to(device)
y = layer(x)
print(f"Output shape: {y.shape}, device: {y.device}")

# Test 3: Full BigNet3Bit
print("\n=== Test 3: BigNet3Bit Full Model ===")
print("Loading original BigNet...")
bignet = BigNet()
bignet.load_state_dict(torch.load('bignet.pth', weights_only=True))
bignet = bignet.to(device)

print("Loading BigNet3Bit...")
bignet3 = BigNet3Bit()
bignet3.load_state_dict(torch.load('bignet.pth', weights_only=True))
bignet3 = bignet3.to(device)

print("Testing forward pass...")
x = torch.randn(10, 4).to(device)
with torch.no_grad():
    y_original = bignet(x)
    y_quantized = bignet3(x)

diff = (y_original - y_quantized).abs()
print(f"\nForward pass difference:")
print(f"  Max: {diff.max():.6f}")
print(f"  Mean: {diff.mean():.6f}")

if diff.max() > 0.5:
    print(f"\n❌ FAILED - Max difference too high: {diff.max():.4f}")
    print("This would fail the online grader!")
else:
    print(f"\n✅ PASSED - Max difference acceptable: {diff.max():.4f}")
    print("This should pass the online grader!")

# Test 4: Memory usage
print("\n=== Test 4: Memory Usage ===")
from homework.stats import model_size
size_mb = model_size(bignet3)
print(f"BigNet3Bit size: {size_mb:.2f} MB")
if size_mb < 9.0:
    print(f"✅ Memory requirement met (< 9 MB)")
else:
    print(f"❌ Memory too high: {size_mb:.2f} MB >= 9 MB")
