#include <chrono> // for timer
#include <cassert>
#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "L1_min.hpp"
using namespace std;

float  L1_norm_cuda(const Eigen::MatrixXf &A, const Eigen::VectorXf &b, Eigen::VectorXf &x,
                   int &k, const int MAX_ITER, const float tol);


uint64_t getCurrTime()
{
    return chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now().        time_since_epoch()).count();
}

int main(int argc, char *argv[]) {
    
    assert((argc >= 2) && "Please supply M and N as the argument");
    
    string filename = "GPU_vs_CPU_result.csv";
    fstream fs;
    fs.open (filename,  fstream::out | fstream::app);

    // Create A (m by n) and b (m by 1) on host and device.
    int m, n;
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    // push to file
    fs << m <<","<< n << ",";
    cout << "M = " << m << ", N = " << n << endl;
    const int MAX_ITER = 1000;
    float TOL;
    if(argc == 4){
        TOL = atof(argv[3]);
        cout << "The input tolerance is " << TOL << endl;
    }else{
        TOL = 1.0e-6 * n;// without tolerance arugement
    }
    const Eigen::MatrixXf A = Eigen::MatrixXf::Random(m, n);
    const Eigen::VectorXf X_GT = Eigen::VectorXf::Random(n); // Ground truth of X
    Eigen::VectorXf b = A * X_GT;
    //Corrupt by flipping signs (outliers) of 33%
    for (int i = 0; i < n/3; ++i) {b[i] = -b[i];}
         
    // Derive an L1 solution: minimize_x ||Ax - b||_1
    Eigen::VectorXf x_L1;
    cout << "Solving using CUDA:" << endl;
    int itr = 0; // the iteration in the loop
    uint64_t tick = getCurrTime();
    float l1_residual = L1_norm_cuda(A, b, x_L1, itr, MAX_ITER, TOL);
    uint64_t tock = getCurrTime();
    float elapsedTime = (tock - tick) / 1000.0f;
    //push to file
    fs << elapsedTime <<"," << itr << ",";

    cout << "Elaposed Time of L1-norm solution: " << elapsedTime << " ms" << endl;
    
    cout << "Difference between L1 solution and the ground truth" << endl;
    //cout << x_L1 - X_GT << endl; // Output the difference from the ground truth
    cout << "Residual: " << l1_residual << endl;
    cout << "--------------------------------" << endl;
    
    cout << "Sovling using Eigen on CPU:" <<endl;
    tick = getCurrTime();
    l1_residual = L1_residual_min(A, b, x_L1, itr, MAX_ITER, TOL);
    tock = getCurrTime();
    
    elapsedTime = (tock - tick) / 1000.0f;
    fs << elapsedTime << "," << itr << endl;
    cout << "Elaposed Time of L1-norm solution: " << elapsedTime << " ms" << endl;
    
    cout << "Difference between L1 solution and the ground truth" << endl;
    //cout << x_L1 - X_GT << endl; // Output the difference from the ground truth
    cout << "Residual: " << l1_residual << endl;
    cout << "------------------------------" << endl;
}


