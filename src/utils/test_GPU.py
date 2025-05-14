import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
print("Num GPUs Available:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    
# PyTorch version: 2.4.1+cu124
# CUDA available: True
# CUDA version: 12.4
# Num GPUs Available: 1
# GPU Name: NVIDIA GeForce RTX 4070 Laptop GPU