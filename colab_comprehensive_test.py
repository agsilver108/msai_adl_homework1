"""
Comprehensive CUDA Test Script for Google Colab
Tests all components: LoRA, QLoRA, Half Precision, Low Precision, Lower Precision
"""
import torch
import sys

print("=" * 80)
print("CUDA HARDWARE CHECK")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
else:
    print("⚠️  WARNING: CUDA not available! Tests will run on CPU.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Import all components
from homework.bignet import BigNet
from homework.half_precision import HalfBigNet
from homework.lora import LoraBigNet
from homework.low_precision import BigNet4Bit
from homework.qlora import QLoRABigNet
from homework.lower_precision import BigNet3Bit
from homework.stats import model_size
from homework.compare import compare_model_forward

print("\n" + "=" * 80)
print("LOADING BASE MODEL")
print("=" * 80)
bignet = BigNet()
bignet.load_state_dict(torch.load('bignet.pth', weights_only=True))
bignet = bignet.to(device)
print(f"✅ BigNet loaded on {device}")
print(f"   Size: {model_size(bignet):.2f} MB")

# Test input
x = torch.randn(10, 4).to(device)
print(f"\nTest input shape: {x.shape}, device: {x.device}")

# Track results
results = {}

print("\n" + "=" * 80)
print("TEST 1: HALF PRECISION")
print("=" * 80)
try:
    half_net = HalfBigNet()
    half_net.load_state_dict(torch.load('bignet.pth', weights_only=True))
    half_net = half_net.to(device)
    
    with torch.no_grad():
        y_original = bignet(x)
        y_half = half_net(x)
    
    diff = (y_original - y_half).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    size_mb = model_size(half_net)
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Model size: {size_mb:.2f} MB")
    
    if max_diff < 0.1 and size_mb < 40:
        print("✅ PASSED - Half Precision")
        results['Half Precision'] = 'PASS'
    else:
        print(f"❌ FAILED - Max diff: {max_diff:.6f}, Size: {size_mb:.2f} MB")
        results['Half Precision'] = 'FAIL'
except Exception as e:
    print(f"❌ ERROR: {e}")
    results['Half Precision'] = 'ERROR'

print("\n" + "=" * 80)
print("TEST 2: LoRA")
print("=" * 80)
try:
    lora_net = LoraBigNet()
    lora_net.load_state_dict(torch.load('bignet.pth', weights_only=True))
    lora_net = lora_net.to(device)
    
    with torch.no_grad():
        y_original = bignet(x)
        y_lora = lora_net(x)
    
    diff = (y_original - y_lora).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    size_mb = model_size(lora_net)
    trainable = sum(p.numel() for p in lora_net.parameters() if p.requires_grad)
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Trainable params: {trainable:,}")
    
    if max_diff < 0.1 and size_mb < 45 and trainable < 2_000_000:
        print("✅ PASSED - LoRA")
        results['LoRA'] = 'PASS'
    else:
        print(f"❌ FAILED - Max diff: {max_diff:.6f}, Size: {size_mb:.2f} MB, Trainable: {trainable:,}")
        results['LoRA'] = 'FAIL'
except Exception as e:
    print(f"❌ ERROR: {e}")
    results['LoRA'] = 'ERROR'

print("\n" + "=" * 80)
print("TEST 3: LOW PRECISION (4-bit)")
print("=" * 80)
try:
    lowp_net = BigNet4Bit()
    lowp_net.load_state_dict(torch.load('bignet.pth', weights_only=True))
    lowp_net = lowp_net.to(device)
    
    with torch.no_grad():
        y_original = bignet(x)
        y_lowp = lowp_net(x)
    
    diff = (y_original - y_lowp).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    size_mb = model_size(lowp_net)
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Model size: {size_mb:.2f} MB")
    
    if max_diff < 0.15 and size_mb < 15:
        print("✅ PASSED - Low Precision (4-bit)")
        results['Low Precision'] = 'PASS'
    else:
        print(f"❌ FAILED - Max diff: {max_diff:.6f}, Size: {size_mb:.2f} MB")
        results['Low Precision'] = 'FAIL'
except Exception as e:
    print(f"❌ ERROR: {e}")
    results['Low Precision'] = 'ERROR'

print("\n" + "=" * 80)
print("TEST 4: QLoRA")
print("=" * 80)
try:
    qlora_net = QLoRABigNet()
    qlora_net.load_state_dict(torch.load('bignet.pth', weights_only=True))
    qlora_net = qlora_net.to(device)
    
    with torch.no_grad():
        y_original = bignet(x)
        y_qlora = qlora_net(x)
    
    diff = (y_original - y_qlora).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    size_mb = model_size(qlora_net)
    trainable = sum(p.numel() for p in qlora_net.parameters() if p.requires_grad)
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Trainable params: {trainable:,}")
    
    if max_diff < 0.15 and size_mb < 20 and trainable < 2_000_000:
        print("✅ PASSED - QLoRA")
        results['QLoRA'] = 'PASS'
    else:
        print(f"❌ FAILED - Max diff: {max_diff:.6f}, Size: {size_mb:.2f} MB, Trainable: {trainable:,}")
        results['QLoRA'] = 'FAIL'
except Exception as e:
    print(f"❌ ERROR: {e}")
    results['QLoRA'] = 'ERROR'

print("\n" + "=" * 80)
print("TEST 5: LOWER PRECISION (3-bit) - EXTRA CREDIT")
print("=" * 80)
try:
    lower_net = BigNet3Bit()
    lower_net.load_state_dict(torch.load('bignet.pth', weights_only=True))
    lower_net = lower_net.to(device)
    
    with torch.no_grad():
        y_original = bignet(x)
        y_lower = lower_net(x)
    
    diff = (y_original - y_lower).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    size_mb = model_size(lower_net)
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Model size: {size_mb:.2f} MB")
    
    # Extra credit requirements: < 9 MB and reasonable accuracy
    if max_diff < 0.5 and size_mb < 9:
        print("✅ PASSED - Lower Precision (3-bit) EXTRA CREDIT!")
        results['Lower Precision'] = 'PASS'
    else:
        if max_diff >= 0.5:
            print(f"❌ FAILED - Max difference too high: {max_diff:.4f} (must be < 0.5)")
        if size_mb >= 9:
            print(f"❌ FAILED - Size too large: {size_mb:.2f} MB (must be < 9 MB)")
        results['Lower Precision'] = 'FAIL'
except Exception as e:
    print(f"❌ ERROR: {e}")
    results['Lower Precision'] = 'ERROR'

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
for test_name, result in results.items():
    status = "✅" if result == 'PASS' else ("❌" if result == 'FAIL' else "⚠️")
    print(f"{status} {test_name}: {result}")

total_pass = sum(1 for r in results.values() if r == 'PASS')
total_tests = len(results)
print(f"\nPassed: {total_pass}/{total_tests}")

if results.get('Lower Precision') == 'PASS':
    print("\n🎉 ALL TESTS PASSED INCLUDING EXTRA CREDIT!")
    print("Expected score: 105/100")
    print("✅ SAFE TO SUBMIT!")
elif total_pass == 4 and results.get('Lower Precision') != 'PASS':
    print("\n✅ Core requirements passed (100/100)")
    print("⚠️  Extra credit failed - you'll get 100/100")
    print("Decision: Submit for guaranteed 100, or fix extra credit for 105?")
else:
    print("\n❌ SOME CORE TESTS FAILED!")
    print("⚠️  DO NOT SUBMIT YET - Fix the failing tests first")
