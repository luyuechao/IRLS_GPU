#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>    //defines operator overriding
#include "cublas_v2.h"
#include "Eigen/Dense"
#include "magma_v2.h"
#include "magma_lapack.h"

using namespace std;
const float EPS = 1.0e-8;   // A constant small number (used for avoiding zero-division)

void update_W(float* W, const int M, const float* e);

float L1_norm_cuda(const Eigen::MatrixXf &A, const Eigen::VectorXf &b, Eigen::VectorXf &x,
                   int &k, const int MAX_ITER, const float TOL){
    cout << "tol = " << TOL << endl;
   /* initilize MAGMA */ 
    magma_init();
    magma_print_environment();
    
    float gpu_error, cpu_error, error, Anorm, work[1];
    float c_one = MAGMA_S_ONE;
    float  c_neg_one = MAGMA_S_NEG_ONE;
    float *h_A, *h_A2, *h_B, *h_X, *h_R, *tau, *h_work, tmp[1];
    magmaFloat_ptr d_A, d_B;
    magma_int_t M, N, size, nrhs, lda, ldb, ldda, lddb, min_mn, max_mn, nb, info;
    magma_int_t lworkgpu, lhwork, lhwork2;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    int status = 0;
    magma_queue_t queue1 = NULL;
    int dev = 0;
    magma_queue_create( dev, &queue1 );
    nrhs = 1;//number of right hand side
    
    const float EPS = 1.0e-8;      // A constant small number (used for avoiding zero-division)
    float residual = std::numeric_limits<float>::infinity(); // Set the initial value for the residual to be infinity

    M = A.rows();     // # of rows of Matrix A 
    N = A.cols();     // # of cols of Matrix A

    min_mn = fmin(M, N);
    max_mn = fmax(M, N);
    lda    = M;
    ldb    = max_mn;
    size   = lda * N;
    ldda   = magma_roundup( M, 32 );  // multiple of 32 by default
    lddb   = magma_roundup( max_mn, 32 );  // multiple of 32 by default
    nb     = magma_get_sgeqrf_nb( M, N );
    lworkgpu = (M - N + nb) * (nrhs + nb) + nrhs * nb;

    if (M != b.size())  // M should agree with the size of Vector b (otherwise, throw an exception)
        {throw std::invalid_argument("Exception: Inconsistent dimensionality");}

    Eigen::DiagonalMatrix<float, Eigen::Dynamic> W(M);  // Prepare an (M x M) diagonal weight matrix
    W.setIdentity();    // Initialize the diagonal matrix as identity
    Eigen::VectorXf xold = 1000.0f * Eigen::VectorXf::Ones(N);// Buffer for storing previous solution x (randomly initialize)
    float *host_A, *host_b, *host_d, *host_W, *host_C, *host_x, *host_x_old;
    checkCudaErrors(cudaHostAlloc((void**)&host_A,     M * N * sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&host_b,     M *     sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&host_d,     M *     sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&host_W,     M * M * sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&host_C,     M * M * sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&host_x,         M * sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&host_x_old,     M * sizeof(float), cudaHostAllocDefault));
 
    // copy eigen data to raw c++ data
    Eigen::Map<Eigen::MatrixXf>(host_A, M, N) = A;
    Eigen::Map<Eigen::VectorXf>(host_b, M) = b;
    Eigen::Map<Eigen::VectorXf>(host_x_old, N) = xold;
    Eigen::Map<Eigen::MatrixXf>(host_W, M, M) = W;
    
    // allocate device memory
    float *dev_A, *dev_b, *dev_x, *dev_x_old, *dev_W, *dev_C, *dev_d;
    checkCudaErrors(cudaMalloc(&dev_A, M * N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&dev_b, M *     sizeof(float)));
    checkCudaErrors(cudaMalloc(&dev_W, M * M * sizeof(float)));
    checkCudaErrors(cudaMalloc(&dev_x,     M * sizeof(float)));
    checkCudaErrors(cudaMalloc(&dev_x_old, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&dev_C, M * N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&dev_d, M *     sizeof(float)));

    checkCudaErrors(cudaMemcpy(dev_A, host_A, M * N *     sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_b, host_b, M *         sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_W, host_W, M * M *     sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_x_old, host_x_old, N * sizeof(float), cudaMemcpyHostToDevice));

    // Initilize CUBLAS    
    cublasStatus_t cublasStat;
    cublasHandle_t cublasH = 0;
    cublasCreate(&cublasH);
    //Initilize 
        // query for workspace size
    lhwork = -1;
    lapackf77_sgeqrf(&M, &N, NULL, &M, NULL, tmp, &lhwork, &info);
    lhwork2 = (magma_int_t) MAGMA_S_REAL( tmp[0] );
    
    lhwork = -1;
    lapackf77_sormqr( MagmaLeftStr, MagmaTransStr,
                     &M, &nrhs, &min_mn, NULL, &lda, NULL,
                     NULL, &ldb, tmp, &lhwork, &info);
    lhwork = (magma_int_t) MAGMA_S_REAL( tmp[0] );
    lhwork = fmax( fmax( lhwork, lhwork2 ), lworkgpu );
    //cout << " size of lhwork = " << lhwork << endl;//for debug
    
    magma_smalloc_cpu( &h_work, lhwork );
    
    const float alpha = 1.0f, beta = 0.0f, alpha2 = -1.0f; 
    float norm_result;
    
    for ( k = 0; k < MAX_ITER; ++k){ // Start iteration of IRLS
    
      // C = W * A;  W:[M, M], A:[M, N], C:[M, N]
      cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, M, N, M,
                  &alpha, dev_W, M, dev_A, M, &beta, dev_C, M);

     // d = W * b;
      cublasSgemv(cublasH, CUBLAS_OP_N, M, M,
                 &alpha, dev_W, M, dev_b, 1, &beta, dev_d, 1);
      
      cudaDeviceSynchronize();
      // Derive an approximate solution of Cx = d for x via least squares
      // Calculate the size of work buffer needed.

      // Solve Ax = b for x with MAGMA, x is n x 1.
      magma_sgels3_gpu( MagmaNoTrans, M, N, nrhs, dev_C, M,
                        dev_d, M, h_work, lworkgpu, &info);
      
      if(info != 0){
        cout <<"magma_sgels3_gpu returned error " << info << magma_strerror(info);
      }
      
      // Copy the x from d
      checkCudaErrors(cudaMemcpy(dev_x, dev_d, N * sizeof(float), cudaMemcpyDeviceToDevice));
      
      cudaDeviceSynchronize();
      // x_old = -x + x_old
      cublasSaxpy(cublasH, N, &alpha2, dev_x, 1, dev_x_old, 1);
            
      // residual computed as L1 norm of residual vector e
      cublasSnrm2(cublasH, N, dev_x_old, 1, &norm_result);

      if(norm_result < TOL){
          cout << "norm_result = " << norm_result << endl; 
          cout << "Converage at " << k << " iterations:" << endl; break;
      }

      // x_old = x;
      checkCudaErrors(cudaMemcpy(dev_x_old, dev_x, N * sizeof(float), cudaMemcpyDeviceToDevice));
      //Residual vector e = A * x - b
      checkCudaErrors(cudaMemcpy(dev_d, dev_b, M * sizeof(float), cudaMemcpyDeviceToDevice));
      cublasSgemv(cublasH, CUBLAS_OP_N, M, N, &alpha, dev_A, 
                                  M, dev_x, 1, &alpha2, dev_d, 1);
      //residual computed as L1 norm of residual vector e
      cublasSasum(cublasH, N, dev_d, 1, &residual);
      checkCudaErrors(cudaMemcpy(host_b, dev_d, M * sizeof(float), cudaMemcpyDeviceToDevice));
      
      // Update weight matrix W based on e (dev_d holds e)
      update_W(dev_W, M, dev_d);
      getLastCudaError("Weight update kernel failed\n");
    }
    
    cout << "iterations k = " << k << endl;
    checkCudaErrors(cudaMemcpy(host_b, dev_x, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    x = Eigen::Map<Eigen::VectorXf>(host_b, N);

    cublasDestroy(cublasH);
    cudaFree(dev_A);
    cudaFree(dev_b);
    cudaFree(tau);
    cudaFree(work);

    return residual;
}                                                                                
