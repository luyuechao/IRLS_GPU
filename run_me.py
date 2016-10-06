#!/usr/bin/env python
import subprocess

subprocess.call(["rm", "GPU_vs_CPU_result.csv"])

program = './build/IRLS_CUDA'
a = 128
b = 32
for i in range(1,16):
    a = a * 1.5
    b = b * 1.5
    args=[program, str(a), str(b), '&']
    subprocess.call(args)
