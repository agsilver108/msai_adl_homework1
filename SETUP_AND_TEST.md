# HW1 Deep Learning - Setup and Testing Guide

Complete guide to set up and test the memory-efficient neural networks assignment on a new machine.

## Prerequisites

- Python 3.10+ (CPU-only PyTorch compatible)
- Git
- GitHub account access to: https://github.com/agsilver108/msai_adl_homework1

## Quick Setup

### 1. Clone Repository

```bash
git clone https://github.com/agsilver108/msai_adl_homework1
cd msai_adl_homework1
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If using Python 3.14, CUDA-enabled PyTorch is not available. Use CPU version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Local Testing

### Test with Grader

```bash
python -m grader homework
```

**Expected Output:**
```
Total                                                    105 / 100
```

### Create Submission Bundle

```bash
python bundle.py homework agsil
```

This creates `agsil.zip` containing all homework files.

### Test Bundle

```bash
python -m grader agsil.zip
```

### Multiple Test Runs (Check Consistency)

```bash
python test_cuda_repeat.py
```

This runs the grader 10 times and reports:
- All scores
- Min/Max/Average scores

**Expected:** Consistent 100-105 scores across all runs.

## Colab T4 GPU Testing

### 1. Upload to Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `test_colab_t4.ipynb`
3. Change runtime to GPU (Runtime → Change runtime type → T4 GPU)

### 2. Upload Bundle

Upload `agsil.zip` to Colab using the file browser.

### 3. Run All Sections

Execute all notebook sections in order:
- Section 1: Setup Environment
- Section 2: Upload and Extract Submission
- Section 3-8: Individual component tests
- Section 9: Summary
- Section 10: Run Official Grader

### 4. Verify Results

**Expected output from Section 10:**
```
Total                                                    105 / 100
```

All custom tests should show `✓ PASS`:
- 4-bit Quantization
- QLoRA
- 3-bit Extra Credit
- Buffer Device Handling (Critical CUDA test)

## Implementation Details

### Core Files

1. **homework/half_precision.py** - Float16 compression (20/20 pts)
   - Memory: 36.07 MB
   - Simple dtype conversion

2. **homework/lora.py** - Low-rank adapters (30/30 pts)
   - Memory: 40.57 MB
   - 1.19M trainable parameters

3. **homework/low_precision.py** - 4-bit quantization (20/20 pts)
   - Memory: 11.36 MB
   - Block-based symmetric quantization

4. **homework/qlora.py** - 4-bit + LoRA (30/30 pts)
   - Memory: 15.86 MB
   - Combines quantization and adaptation

5. **homework/lower_precision.py** - 3-bit quantization (5/5 pts extra credit)
   - Memory: 7.98 MB
   - Asymmetric quantization with group_size=64

### Key Implementation Features

**Buffer Device Handling:**
```python
# In _load_state_dict_pre_hook, use existing buffer device (set by model.to)
# Don't use weight.device as it's from CPU state_dict being loaded
target_device = self.weight_q4.device
self.weight_q4.data = packed.view(-1).to(target_device)
self.weight_norm.data = norm.to(target_device)
```

**3-bit Asymmetric Quantization:**
- Stores min/max per group instead of just normalization
- group_size=64 for better accuracy
- Packs 8 values into 3 bytes

**QLoRA Fix:**
```python
# Reshape weight_q4 from 1D to 2D before dequantization
weight_q4_2d = self.weight_q4.view(num_groups, -1)
weight_dequant = block_dequantize_4bit(weight_q4_2d, self.weight_norm)
```

## Troubleshooting

### Issue: Device Mismatch Errors on CUDA

**Symptom:** Buffers stay on CPU after loading state_dict
**Solution:** Use `self.weight_q4.device` (existing buffer device) not `weight.device` (CPU)

### Issue: QLoRA Crashes

**Symptom:** Shape mismatch in block_dequantize_4bit
**Solution:** Reshape weight_q4 to 2D before dequantization

### Issue: 3-bit Extra Credit Fails

**Symptom:** Max difference > 0.5 or memory > 9 MB
**Solution:** Use asymmetric quantization with group_size=64

### Issue: Inconsistent Scores

**Symptom:** Scores vary between 95-105
**Solution:** Normal - some randomness in grader tests. Should average 100+

## Submission Checklist

- [ ] Local grader shows 105/100
- [ ] Bundle created: agsil.zip
- [ ] Bundle tested: `python -m grader agsil.zip` shows 105/100
- [ ] Colab T4 test passes all sections
- [ ] Buffer device handling test passes on Colab
- [ ] Multiple test runs show consistent 100-105 scores
- [ ] GitHub repo updated with latest code
- [ ] Submit agsil.zip to Canvas

## Final Verification

Run this command sequence before final submission:

```bash
# Bundle and test
python bundle.py homework agsil
python -m grader agsil.zip

# Multiple runs for consistency
python test_cuda_repeat.py

# Push to GitHub
git add .
git commit -m "Final submission ready"
git push
```

Then test on Colab one final time before Canvas submission.

## Repository Structure

```
msai_adl_homework1/
├── homework/
│   ├── __init__.py
│   ├── bignet.py           # Base model
│   ├── half_precision.py   # Float16 (20 pts)
│   ├── lora.py             # LoRA (30 pts)
│   ├── low_precision.py    # 4-bit (20 pts)
│   ├── qlora.py            # QLoRA (30 pts)
│   ├── lower_precision.py  # 3-bit extra credit (5 pts)
│   ├── compare.py          # Comparison utilities
│   ├── fit.py              # Training utilities
│   └── stats.py            # Profiling
├── grader/
│   ├── __main__.py
│   ├── grader.py
│   └── tests.py
├── bignet.pth              # Baseline model weights
├── bundle.py               # Create submission zip
├── requirements.txt        # Dependencies
├── test_colab_t4.ipynb    # Comprehensive Colab test
├── test_cuda_repeat.py    # Multiple run testing
└── README.md

Score Breakdown:
- LoRA: 30/30
- QLoRA: 30/30
- Half Precision: 20/20
- 4-bit Quantization: 20/20
- 3-bit Extra Credit: 5/5
Total: 105/100
```

## Contact

If issues persist, verify:
1. Latest code from GitHub main branch
2. Python version compatibility (3.10-3.13 recommended)
3. Virtual environment activated
4. Dependencies installed correctly
5. Colab using T4 GPU runtime

Last updated: February 7, 2026
