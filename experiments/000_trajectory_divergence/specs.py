import torch
import psutil

# on macOS so using Metal Performance Shader (MPS)
# instead of CUDA which is used on NVIDIA
is_mps_available = torch.backends.mps.is_available()

print(is_mps_available)
if is_mps_available:
    print("MPS device (Apple Silicon GPU), unified memory:",
          round(psutil.virtual_memory().total / (1024**3)),
          "GB"
    )
