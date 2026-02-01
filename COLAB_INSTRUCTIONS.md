# Google Colab Testing Instructions

## Step 1: In Colab, clone your GitHub repo

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

## Step 2: Install PyTorch with CUDA (if needed)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
```

## Step 3: Run the CUDA test

```python
!python colab_test.py
```

OR paste this directly in a Colab cell:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from homework.lower_precision import BigNet3Bit
from homework.bignet import BigNet

# Load models
bignet = BigNet()
bignet.load_state_dict(torch.load('bignet.pth', weights_only=True))
bignet = bignet.to(device)

bignet3 = BigNet3Bit()
bignet3.load_state_dict(torch.load('bignet.pth', weights_only=True))
bignet3 = bignet3.to(device)

# Test
x = torch.randn(10, 4).to(device)
with torch.no_grad():
    y1 = bignet(x)
    y2 = bignet3(x)

diff = (y1 - y2).abs()
print(f"Max diff: {diff.max():.4f}")
print(f"Mean diff: {diff.mean():.4f}")

if diff.max() > 0.5:
    print("❌ FAILED - would not pass online grader")
else:
    print("✅ PASSED - should pass online grader")

# Check memory
from homework.stats import model_size
size_mb = model_size(bignet3)
print(f"Size: {size_mb:.2f} MB (must be < 9 MB)")
```

## What you're looking for:
- ✅ Max diff < 0.5
- ✅ Size < 9 MB
- If both pass → Submit to Canvas
- If either fails → Come back and we'll fix it
