#!/usr/bin/env python
import subprocess

subprocess.call(["rm", "GPU_vs_CPU_result.csv"])

program = './build/IRLS_CUDA'
M = 128
N = 32
tolerance = 1.0e-8 * N

for i in range(1,5):
    args=[program, str(M), str(N),str(tolerance)]
    subprocess.call(args)
    M = M * 2
    N = N * 2

print ("finished testing, check the result in GPU_vs_CPU_result.csv")
