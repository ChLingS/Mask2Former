import os
import torch
import subprocess

print("="*50)
print("PyTorch 信息:")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"设备 {i}: {torch.cuda.get_device_name(i)}")

print("\n" + "="*50)
print("系统信息:")
try:
    print("NVIDIA 驱动信息:")
    print(subprocess.check_output("nvidia-smi", shell=True).decode())
except Exception as e:
    print(f"无法执行 nvidia-smi: {e}")

try:
    print("CUDA 编译器版本:")
    print(subprocess.check_output("nvcc --version", shell=True).decode())
except Exception as e:
    print(f"无法执行 nvcc: {e}")

print("\n" + "="*50)
print("环境变量:")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '未设置')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '未设置')}")