#!/usr/bin/env python
import subprocess

subprocess.call(["rm", "GPU_vs_CPU_result.csv"])

program = './build/IRLS_CUDA'
a = 128
b = 8
for i in range(1,50):
    args=[program, str(a), str(b), '&']
    subprocess.call(args)
    a = a * 2
    b = b * 2

