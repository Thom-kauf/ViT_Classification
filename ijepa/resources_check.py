import os
import psutil
import torch

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f'Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB')

# Before loading
print_memory_usage()

checkpoint = torch.load('/home/tomtom/ijepa/checkpoint/IN1K-vit.h.14-300e.pth.tar', map_location=torch.device('cpu'))

# After loading
print_memory_usage()
