# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
verify_gpu.py
Quick sanity check to confirm PyTorch can see the RTX 5080
and that CUDA operations work correctly before training begins.
"""

import torch

print("=" * 60)
print("  PyTorch GPU Verification")
print("=" * 60)
print(f"PyTorch version     : {torch.__version__}")
print(f"CUDA available      : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version        : {torch.version.cuda}")
    print(f"GPU name            : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability  : sm_{cap[0]}{cap[1]}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM          : {vram:.1f} GB")

    # Test a real CUDA operation — matrix multiply on GPU
    print("\nRunning CUDA matrix multiply test...")
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    c = torch.matmul(a, b)
    print(f"Matrix result shape : {c.shape}")
    print(f"Result device       : {c.device}")

    # Test FP16 mixed precision — this is what training will use
    print("\nTesting FP16 mixed precision (GradScaler)...")
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
    with autocast():
        d = torch.matmul(a.half(), b.half())
    print(f"FP16 result dtype   : {d.dtype}")
    print(f"FP16 result shape   : {d.shape}")

    print("\nAll GPU checks PASSED. RTX 5080 is ready for training.")
else:
    print("ERROR: CUDA not available. Check PyTorch install.")
    print("Expected: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")

print("=" * 60)

