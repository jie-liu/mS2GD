/*
====================================================================================
           Utility Functions for Sparse Data (CSC)
====================================================================================
  Author: Jie Liu
  Website: http://coral.ie.lehigh.edu/~jild13/
  Last Update: Nov 24, 2014
====================================================================================
  Description:
      "logistic_L2.h" contains functions that are necessary in the
  Min-Batch Proximal Stochastic Variance Reduced Gradient Method for
  logistic regression problem. Usage for each function is available
  immediately prior to the function.
------------------------------------------------------------------------------------
  Input (same for all functions):
        *Xt     ~ double, sparse data matrix
        *x      ~ double, a data point from sparse data *Xt,
        *w      ~ double, current iterate, (d x 1)
        *Y      ~ double, labels from data, (d x 1)
         y      ~ double, a label from labels *Y (corresponds to *x)
        *jc     ~ long, indexes of first non-zero elements of
                 columns of the data matrix, ((n+1) x 1)
        *ir     ~ long, row indexes of elements of the data matrix
        *g      ~ double, full gradient of logistic regression
        *wk     ~ double, previous inner iterate, (d x 1)
        *wtilde ~ double, initial inner iterate in outer iteration,
                 (d x 1)
        *At     ~ long, samples from reservoir sampling (k,n)
         d      ~ long, number of features for each data point
         n      ~ long, number of data points
         b      ~ long, number of mini-batches
         k      ~ long, parameter of reservoir sampling, equals b
====================================================================================
  Copyright by Jie Liu, H.S. Mohler Laboratory, Lehigh University.
====================================================================================
 */

#include <vector>
#include <math.h>
using namespace std;

//template <class R>

/*
====================================================================================
           Compute Function Value for Sparse Data (CSC)
====================================================================================
  Author: Jie Liu
  Website: http://coral.ie.lehigh.edu/~jild13/
  Last Update: Nov 18, 2014
====================================================================================
  Usage:
        compute_fun_val_sparse(double *Xt, double *w, double *Y,
		long n, long *jc, long *ir)
------------------------------------------------------------------------------------
  Input: Check inputs at the beginning of the "logistic_L2.h".
------------------------------------------------------------------------------------
  Return:
        fun_val ~ double, function value, i.e. sum(log(1+(-w.*x*y))/n
====================================================================================
 */
template <class T>
T compute_fun_val_sparse(T *Xt, std::vector<T> &w, T *Y, long n, long *jc, long *ir,
                         long d, T lambda)
{
	T fun_val = 0;
	T temp;
	for(unsigned long i=0;i<n;i++){
		temp = 0;
		for(unsigned long j=jc[i]; j<jc[i+1];j++){
			temp += Xt[j]*w[ir[j]];
		}
		fun_val += log(1+exp(-temp*Y[i]));
	}

	fun_val = fun_val/n;
    
    // Add regularization
    for (unsigned long j=0;j<d;j++){
        fun_val += 0.5*lambda*w[j]*w[j];
    }

	return fun_val;
}



/*
====================================================================================
      Compute Function Gradient(Single) for Sparse Data (CSC)
====================================================================================
  Author: Jie Liu
  Website: http://coral.ie.lehigh.edu/~jild13/
  Last Update: Nov 18, 2014
====================================================================================
  Usage:
        compute_fun_grad_sparse(double *x, double *w, double y,
		long d, long *ir)
------------------------------------------------------------------------------------
  Input: Check inputs at the beginning of the "logistic_L2.h".
------------------------------------------------------------------------------------
  Return:
        temp ~ double, function gradient multiple, i.e.
             -y/(1+exp(w.*x*y))
====================================================================================
 */
template <class T>
T compute_fun_grad_sparse(T *x, std::vector<T> &w, T y, long d, long *ir)
{
	T temp = 0;

	for(unsigned long i=0;i<d;i++){
		temp += w[ir[i]]*x[i];
	}
	temp = exp(y*temp);
	temp = -y/(1+temp);

	return temp;
}


/*
====================================================================================
       Compute Function Full Gradient for Sparse Data (CSS)
====================================================================================
  Author: Jie Liu
  Website: http://coral.ie.lehigh.edu/~jild13/
  Last Update: Nov 18, 2014
====================================================================================
  Usage:
        compute_fun_fullgrad_sparse(double *Xt, double *w, double *Y,
		long n, long d, long *jc, long *ir, double *g)
------------------------------------------------------------------------------------
  Input: Check inputs at the beginning of the "logistic_L2.h".
====================================================================================
 */
template <class T>
void compute_fun_fullgrad_sparse(T *Xt, std::vector<T> &w, T *Y, long n,
                long d, long *jc, long *ir, std::vector<T> &g)
{
	T single_grad;
    T nInverse = 1.0/n;
	for(unsigned long i=0;i<n;i++){
		single_grad = compute_fun_grad_sparse<T>(Xt+jc[i], w, Y[i],
				jc[i+1]-jc[i],ir+jc[i]);

		for(unsigned long j=jc[i];j<jc[i+1];j++){
			g[ir[j]] += single_grad*Xt[j];
		}
	}

	for (unsigned long k=0;k<d;k++){
		g[k] *= nInverse;
	}

}

/*
====================================================================================
       Peform Lazy Updates for Sparse Data (CSC)
====================================================================================
  Author: Jie Liu
  Website: http://coral.ie.lehigh.edu/~jild13/
  Last Update: Nov 24, 2014
====================================================================================
  Introduction:
        Lazy updates for mS2GD, S2GD, SGD_PVRG, SAG, SGD
====================================================================================
 */



template <class T>
void lazy_update_mS2GD(std::vector<T> &wk, std::vector<long> &nz,
                       std::vector<T> &vtilde, std::vector<long> &last_seen,
                       T eta, T lambdapar, T lambdapar2, long k, long d)
{
    for (unsigned long ik=0;ik<d;ik++){
        if((!lambdapar2)){
            wk[nz[ik]] -= eta*(k - last_seen[nz[ik]])*vtilde[nz[ik]];
        }
        else {
            wk[nz[ik]] = pow(lambdapar, k-last_seen[nz[ik]])*wk[nz[ik]]
            -lambdapar*lambdapar2*(1-pow(lambdapar, k-last_seen[nz[ik]]))
            *eta*vtilde[nz[ik]];
        }
        last_seen[nz[ik]] = k;
    }
}

template <class T>
void lazy_update_S2GD(std::vector<T> &wk, std::vector<T> &vtilde,
                      std::vector<long> &last_seen, T eta, T lambdapar,
                      T lambdapar2, long k, long *ir, long jc_0, long jc_end)
{
    for (unsigned long ik=jc_0;ik<jc_end;ik++){
        if((!lambdapar2)){
            wk[ir[ik]] -= eta*(k - last_seen[ir[ik]])*vtilde[ir[ik]];
        }
        else {
            wk[ir[ik]] = pow(lambdapar, k-last_seen[ir[ik]])*wk[ir[ik]]
            -lambdapar*lambdapar2*(1-pow(lambdapar, k-last_seen[ir[ik]]))*eta
            *vtilde[ir[ik]];
        }
        last_seen[ir[ik]] = k;
    }
}


template <class T>
void finish_lazy_update_S2GD(std::vector<T> &wk, std::vector<T> &vtilde,
                             std::vector<long> &last_seen, T eta, T lambdapar,
                             T lambdapar2, long k,long d)
{
    for (unsigned long ik=0;ik<d;ik++){
        if(!lambdapar2){
            wk[ik] -= eta*(k - last_seen[ik])*vtilde[ik];
        }else {
            wk[ik] = pow(lambdapar, k-last_seen[ik])*wk[ik]
            -lambdapar*lambdapar2*(1-pow(lambdapar, k-last_seen[ik]))*eta*vtilde[ik];
        }
    }
}



template <class T>
void lazy_update_SGD(std::vector<T> &wk, std::vector<long> &last_seen, T lambdapar,
                     long k, long *ir, long jc_0, long jc_end)
{
    for (unsigned long ik=jc_0;ik<jc_end;ik++){
        wk[ir[ik]] = pow(lambdapar, k-last_seen[ir[ik]])*wk[ir[ik]];
        last_seen[ir[ik]] = k;
    }
}


template <class T>
void finish_lazy_update_SGD(std::vector<T> &wk, std::vector<long> &last_seen,
                            T lambdapar, long k,long d)
{
    for (unsigned long ik=0;ik<d;ik++){
        wk[ik] = pow(lambdapar, k-last_seen[ik])*wk[ik];
    }
}




/*
 ====================================================================================
 Compute Proximal Operator for case with L2 norm
 ====================================================================================
 Author: Jie Liu
 Website: http://coral.ie.lehigh.edu/~jild13/
 Last Update: Nov 18, 2014
 ====================================================================================
 Usage:
 compute_proximal_operator(std::vector<T> &wk, T lambdapar, long d)
 --------------------------------------------------------------------
 Input: Check inputs at the beginning of the "logistic_L2.h".
 ====================================================================================
 */
template <class T>
void compute_proximal_operator(std::vector<T> &wk, T lambdapar, long d){
    for(unsigned i=0;i<d;i++){
        wk[i] = lambdapar*wk[i];   // Proximal operator
    }
}


/*
====================================================================================
                         Reservoir Sampling
====================================================================================
  Author: Jie Liu
  Website: http://coral.ie.lehigh.edu/~jild13/
  Last Update: Nov 18, 2014
====================================================================================
  Description:
      Choose a set of k integers from 1,...,n uniformly at random.
------------------------------------------------------------------------------------
  Usage:
        reservoir_sample(long k, long n, long *At)
------------------------------------------------------------------------------------
  Input: Check inputs at the beginning of the "logistic_L2.h".
------------------------------------------------------------------------------------
  Reference: http://en.wikipedia.org/wiki/Reservoir_sampling .
====================================================================================
 */

void reservoir_sample(long k, long n, std::vector<long> &At)
{
	long j;
	for(unsigned long i=0;i<k;i++){
//		At[i] = i;
		At[i] = rand()/(RAND_MAX+0.0)*n;
	}
//	for(unsigned long i=k;i<n;i++){
//		j = rand()%i;
//
//
//		if(j<k){
//			At[j] = i;
//		}
//	}
}


