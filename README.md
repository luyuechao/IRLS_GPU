# IRLS_CUDA

CUDA implementation of solving L1 norm of Ax-b by Iteratively Reweighted Least-squares (IRLS).

# Requirements
## Hardware
* Nvidia CUDA compatiable GPU (computation ability >= 2.0)
* MAGMA 2.1 does not support Pascal architecture GPU and CUDA 8.0 yet.

## OS
* Ubuntu 14.04 & Mac OS X 10.11 
* Windows has not been tested yet.

# Install

## Dependnecy
* [CMake] (https://cmake.org/download)
* [CUDA 7.0 or 7.5](https://developer.nvidia.com/cuda-downloads)(CUDA 8.0 unsupported yet)
* [MAGMA > 2.1](http://icl.cs.utk.edu/magma/software/index.html)
* [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
* [OpenBLAS](https://github.com/xianyi/OpenBLAS)
* [MKL](https://software.intel.com/en-us/intel-mkl) (Optional)

## Ubuntu 14.04 / Mac OS X 10.11
* Step 1:
Install CUDA, OpenBLAS, MAGMA, Eigen3.

* Step 2:
    1. git clone https://github.com/luyuechao/LRLS_CUDA
    2. In the project directory, edit CMakeList.txt file.
    3. In line 10, add MAGMA installation path to MAGMADIR (e.g. set(MAGMADIR ~/magma-2.1.0).
    4. In line 59, change "arch" and "code" compiler flag according to your GPU architecture.
       Refer to https://en.wikipedia.org/wiki/CUDA
     (e.g. GTX TITAN X Maxwell which has Compute ability 5.2 shall be set "arch" and "code" as arch=compute_52 and code=sm_52)

    ```
 mkdir build && cd build
 cmake ..
 make -j4
 cd .. && ./run_me.py
```

# Performance Comparsion
see [WiKi](https://github.com/luyuechao/IRLS_CUDA/wiki)

