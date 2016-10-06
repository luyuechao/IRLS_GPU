#include <limits>
#include "Eigen/Dense"
#include "Eigen/Sparse"
using namespace std;
typedef Eigen::MatrixXf Mat;    // Dense matrix
typedef Eigen::VectorXf Vec;    // Vector
typedef Eigen::SparseMatrix<float> SpMat;  // Sparse matrix

/**
* Solves Ax = b for x, where A is a dense matrix
**/
void lstsq(const Eigen::MatrixXf &A, const Eigen::VectorXf &b, Eigen::VectorXf &x)
{
    x = A.colPivHouseholderQr().solve(b);
//    x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}
/**
* Solves Ax = b for x, where A is a sparse matrix
**/
void lstsq(const Eigen::SparseMatrix<float> &A, const Eigen::VectorXf &b, Eigen::VectorXf &x)
{
	Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int> > solver;
	solver.compute(A);
	x = solver.solve(b);
}

/**
*
* Solves the following minimization problem
*
*	minimize_x ||Ax - b||_1
*
* using Iteratively Reweighted Least-Squares (IRLS)
*
**/
template <class Matrix>
float L1_residual_min(const Matrix &A, const Eigen::VectorXf &b, Eigen::VectorXf &x, 
                      int &k, const int MAX_ITER, const float tol)
{
	cout << "tol = " << tol << endl;
	const float eps = 1.0e-8;	// A constant small number (used for avoiding zero-division)
	float residual = std::numeric_limits<float>::infinity(); // Set the initial value for the residual to be infinity
	const int m = A.rows();		// # of rows of Matrix A 
	const int n = A.cols();		// # of cols of Matrix A
	if (m != b.size())	// m should agree with the size of Vector b (otherwise, throw an exception)
		throw std::invalid_argument("Exception: Inconsistent dimensionality");

	Eigen::DiagonalMatrix<float, Eigen::Dynamic> W(m);			// Prepare an (m x m) diagonal weight matrix
	W.setIdentity();	// Initialize the diagonal matrix as identity
	Vec xold = 1000.0 * Vec::Ones(n);	// Buffer for storing previous solution x (randomly initialize)
	
        for (k = 0; k < MAX_ITER; ++k)  // Start iteration of IRLS
	{
		// Create a weighted linear system of equations Cx = d (corresponds to WAx = Wb)
		Matrix C = W * A;
		Eigen::VectorXf d = W * b;
		// Derive an approximate solution of Cx = d for x via least squares
		lstsq(C, d, x);
		// Check convergence
		if ((x - xold).norm() < tol) {
                	cout << "Converage at " << k << "iterations." <<endl;
                	break;	// Converged
		}
                
		xold = x;			// Update xold

		Eigen::VectorXf e = b - A * x;	// Compute the residual vector e
		residual = e.lpNorm<1>();		// residual computed as L1 norm of residual vector e
		// Update weight matrix W based on e
		for (int i = 0; i < m; ++i)
		{
			W.diagonal()[i] = 1.0 / fmax(sqrt(fabs(e[i])), eps);	// W(i,i) = 1.0 / sqrt|e(i)|	(max with eps is used for avoiding zero-division in the case e(i) = 0.0)
		}
	}
        cout << "Iterations k = " << k << endl;
	return residual;
}


