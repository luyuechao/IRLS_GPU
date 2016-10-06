#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>    //defines operator overriding
#include "cusolverDn.h"

#define EPS 1.0e-8   // A constant small number (used for avoiding zero-division)

__global__ void update_W_kernel(float* W, const int M, const float* e){

    const int tx = threadIdx.x, bx = blockIdx.x;
    const int diagIdx = bx * blockDim.x + tx;
    float ftemp;

    if(e[diagIdx] < 0){
        ftemp = sqrt(-e[diagIdx]);
    }else if(e[diagIdx] > 0){
        ftemp = sqrt(e[diagIdx]);
    }else{
        ftemp = EPS;
    }
    W[diagIdx * (M + 1)] = 1.0f / fmax(ftemp, EPS);
}

void update_W(float* W, const int M, const float* e){
    
    dim3 blockPerGrid(32, 1);
    dim3 threadsPerBlock(M / 32, 1);
    
    update_W_kernel <<< blockPerGrid, threadsPerBlock >>>(W, M, e);

}
