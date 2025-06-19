# tests/test_gpu.py

import tensorflow as tf
import torch

print("="*50)
print("TensorFlow GPU检查:")
print(f"可用GPU列表: {tf.config.list_physical_devices('GPU')}")
print(f"TensorFlow版本: {tf.__version__}")

print("\n" + "="*50)
print("PyTorch GPU检查:")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
print(f"PyTorch版本: {torch.__version__}")
