

/*
 =================================================================================
   Implementation of mS2GD, S2GD, SGD_PVRG, SAG, SGD, FGD
 =================================================================================
 Author: Jie Liu
 Website: http://coral.ie.lehigh.edu/~jild13/
 Last Update: Oct 24, 2014
 =================================================================================
 Description:
     Implementation of mS2GD, S2GD, SGD_PVRG, SAG, SGD, FGD
 ---------------------------------------------------------------------------------
 Input (same for all functions):
 *Xt     ~ double, sparse data matrix
 *w      ~ double, initial iterate, (d x 1)
 *y      ~ double, labels from data, (d x 1)
 *jc     ~ long, indexes of first non-zero elements of columns of the data matrix, 
          ((n+1) x 1)
 *ir     ~ long, row indexes of elements of the data matrix
  n      ~ long, number of data points
  d      ~ long, number of features for each data point
  T      ~ long, number of outer iterations
  m      ~ long, number of inner loops
  b      ~ long, number of mini-batches
  eta    ~ double, step size
  muF    ~ convexity parameter for function F
  muR    ~ convexity parameter for function R
  lambda ~ regularization parameter for function R
  printheader ~ whether to print (1) header or not (0)
 =================================================================================
 Copyright by Jie Liu, H.S. Mohler Laboratory, Lehigh University.
 ====================================================================================
 */



#include <iostream>
#include <random>
#include <math.h>
#include "logistic_L2.h"
using namespace std;


/* mS2GD */
template<class R>
void mS2GD_PVRG(R *Xt, std::vector<R> &w, R *y, long *jc, long *ir, long n, long d,
                long T, long m, long b, R eta, R muF, R muR, R lambda, int printheader,
                int *seed)
{
    double sparsity = (double)jc[n]/(d*n);   // Compute sparsity of the data
    
    R nInverse = 1.0/n;
    R bInverse= 1.0/b;
    R eta_bInverse = eta*bInverse;
    
    long n_evals=0;
    // Calculate lambdapar for proximal operator
    R lambdapar = 1.0/(1.0+lambda*eta);
    R lambdapar2 = 0;
    if (lambdapar<1){
        lambdapar2 = 1.0/(1.0-lambdapar);
    }
    
    // Calculate gamma and distribution Q;
    double q = (1.0-eta*muF)/(1.0+eta*muR);
    double Q[m];
    for (unsigned i=0;i<m;i++){
        Q[i] = pow(q,m-i-1);
    }
    // Set non-uniform discrete distribution generator, also set randomseeds
    std::default_random_engine generator (seed[0]);
    std::discrete_distribution<long> discrete_dist(Q,Q+m);
    srand(seed[1]);
    
    
    // Print information
    // Print header
    if (printheader){
    cout<<"================================================================="<<endl;
    cout<<"    Method: "<<"mS2GD_PVRG"<<endl;
    cout<<"================================================================="<<endl;
    cout<<"    Parameters:"<<endl;
    cout<<"         n: "<<n<<endl;
    cout<<"         d: "<<d<<endl;
    cout<<"         sparsity: "<<sparsity<<endl;
    cout<<"         m: "<<m<<endl;
    cout<<"         b: "<<b<<endl;
    cout<<"         maxiter: "<<T<<endl;
    cout<<"         eta: "<<eta<<endl;
    cout<<"         lambda: "<<lambda<<endl;
    cout<<"         muF: "<<muF<<endl;
    cout<<"         muR: "<<muR<<endl;
    cout<<"         randomgenerator: "<<seed[0]<<endl;
    cout<<"         randomseed: "<<seed[1]<<endl;
    cout<<"================================================================="<<endl;
    cout<<"  Iter.   Fun.Val.                     Kt.      Eff.Passes"<<endl;
    cout<<"================================================================="<<endl;
    }
    
    // Declare and allocate memory for useful parameters
    R fun_val;
    std::vector<R> wtilde(d);  // wtilde
    std::vector<R> vtilde(d);  // full gradient vtilde
    std::vector<R> wk(d);  // inner iterate
    std::vector<R> vk(d);  //  stochastic gradient vk
    std::vector<long> At(b);  // uniform sampling set At
    std::vector<long> Dc(d);  // dummy variables indicating "non-zero" update
    std::vector<long> nz(d);  // coordinates of "non-zero" update
    std::vector<long> last_seen(d);  // record iterates of last seen in lazy update
    long Kt = 0;  // number of inner loops
    R fi_wk;
    R fi_wtilde;
    
    // Loop indexes
    unsigned long t, k, i, j, ik, nDc;
    
    // Initialize vk
    std::fill(vk.begin(),vk.end(),0);
    
    /* Outer Iteration Starts Here */
    for(t=0;t<T;t++){
        
        // Initialization for wtilde, vtilde and w
        wtilde = w;
        // Set vtilde = 0
        std::fill(vtilde.begin(),vtilde.end(),0);
        std::fill(last_seen.begin(),last_seen.end(),0);
        
        // Evaluate current function value
        fun_val = compute_fun_val_sparse<R>(Xt, wtilde, y, n, jc, ir, d, lambda);
        
        // Print iterates and function values
        printf("  %ld       %+.15e       %ld       %.5e\n", t, fun_val,Kt,n_evals*nInverse);
        
        // Evaluate full gradient of the function
        compute_fun_fullgrad_sparse<R>(Xt, wtilde, y, n, d, jc, ir, vtilde);
        
        // Set wk(w0) <- wtilde
        wk = wtilde;
        
        // Get non-uniform sampling set
        Kt = discrete_dist(generator);
        
        // Set Dc = 0
        std::fill(Dc.begin(),Dc.end(),0);

        
        /* Inner Iteration Starts Here */
        for(k=0;k<Kt;k++){
            
            // Get uniform sampling set
            reservoir_sample(b, n, At);
            
            // Record coodinates of "none-zero" update by dummy variable
            // nDc is the number of coordinates of "non-zero" update
            nDc = 0;
            for(ik=0;ik<b;ik++){
                i = At[ik];
                for(j=jc[i];j<jc[i+1];j++){
                    if (Dc[ir[j]]<k)
                    {
                        Dc[ir[j]] = k;
                        nz[nDc++] = ir[j];
                    }
                }
            }
            
            // Lazy update
            lazy_update_mS2GD<R>(wk, nz, vtilde, last_seen, eta, lambdapar,
                                 lambdapar2, k, nDc);
            
            // Compute first two parts of the stochastic gradient: fi_wk-fi_wtilde
            for(ik=0;ik<b;ik++){
                i = At[ik];
                fi_wk = compute_fun_grad_sparse<R>(Xt+jc[i], wk, y[i],
                        jc[i+1]-jc[i], ir+jc[i]);
                fi_wtilde = compute_fun_grad_sparse<R>(
                        Xt+jc[i], wtilde, y[i], jc[i+1]-jc[i],ir+jc[i]);
                
                for(j=jc[i];j<jc[i+1];j++){
                    vk[ir[j]] += (fi_wk-fi_wtilde)*Xt[j];
                }
            }

            // Update inner iterate
            for (i=0;i<nDc;i++){
                wk[nz[i]] -= eta_bInverse*vk[nz[i]];
                vk[nz[i]] = 0;   // reset vk to 0
            }

        }
        // Finish lazy update
        finish_lazy_update_S2GD(wk, vtilde, last_seen, eta, lambdapar, lambdapar2, Kt, d);
        
        /* Inner Iteration Ends Here */
        n_evals += Kt*b+n;
        
        w = wk;
        
    }

    
    /* Outer Iteration Ends Here */
    
    
    /* Print the Last Iteration */
    // Evaluate current function value
    fun_val = compute_fun_val_sparse<R>(Xt, w, y, n, jc, ir, d, lambda);
    
    // Print the last iteration
    printf("  %ld       %+.15e       %ld       %.5e\n", t, fun_val,Kt,n_evals*nInverse);
    
    
    // Print footer
    if (printheader){
    cout<<"================================================================="<<endl;
    }
    
}

/* S2GD */
template<class R>
void S2GD_PVRG(R *Xt, std::vector<R> &w, R *y, long *jc, long *ir, long n, long d,
               long T, long m, R eta, R muF, R muR, R lambda, int printheader, int *seed)
{
    double sparsity = (double)jc[n]/(d*n);   // Compute sparsity of the data
    R nInverse = 1.0/n;
    
    long n_evals=0;
    // Calculate lambdapar for proximal operator
    R lambdapar = 1.0/(1.0+lambda*eta);
    R lambdapar2 = 0;
    if (lambdapar<1){
        lambdapar2 = 1.0/(1.0-lambdapar);
    }
    
    // Calculate gamma and distribution Q;
    double q = (1.0-eta*muF)/(1.0+eta*muR);
    double Q[m];
    for (unsigned i=0;i<m;i++){
        Q[i] = pow(q,m-i-1);
    }
    // Set non-uniform discrete distribution generator, also set randomseeds
    std::default_random_engine generator (seed[0]);
    std::discrete_distribution<long> discrete_dist(Q,Q+m);
    srand(seed[1]);
    
    
    // Print information
    // Print header
    if (printheader){
    cout<<"================================================================="<<endl;
    cout<<"    Method: "<<"S2GD_PVRG"<<endl;
    cout<<"================================================================="<<endl;
    cout<<"    Parameters:"<<endl;
    cout<<"         n: "<<n<<endl;
    cout<<"         d: "<<d<<endl;
    cout<<"         sparsity: "<<sparsity<<endl;
    cout<<"         m: "<<m<<endl;
    cout<<"         maxiter: "<<T<<endl;
    cout<<"         eta: "<<eta<<endl;
    cout<<"         lambda: "<<lambda<<endl;
    cout<<"         muF: "<<muF<<endl;
    cout<<"         muR: "<<muR<<endl;
    cout<<"         randomgenerator: "<<seed[0]<<endl;
    cout<<"         randomseed: "<<seed[1]<<endl;
    cout<<"================================================================="<<endl;
    cout<<"  Iter.   Fun.Val.                     Kt.      Eff.Passes"<<endl;
    cout<<"================================================================="<<endl;
    }
    
    // Declare and allocate memory for useful parameters
    R fun_val;
    std::vector<R> wtilde(d);  // wtilde
    std::vector<R> vtilde(d);  // full gradient vtilde
    std::vector<R> wk(d);  // inner iterate
    std::vector<long> last_seen(d);  // record iterates of last seen in lazy update
    long Kt = 0;  // number of inner loops
    R fi_wk;
    R fi_wtilde;
    
    // Loop indexes
    unsigned long t, k, j, nnz, ik;
    
    /* Outer Iteration Starts Here */
    for(t=0;t<T;t++){
        
        // Initialization for wtilde, vtilde and w
        wtilde = w;
        // Set vtilde = 0
        std::fill(vtilde.begin(),vtilde.end(),0);
        std::fill(last_seen.begin(),last_seen.end(),0);
        
        // Evaluate current function value
        fun_val = compute_fun_val_sparse<R>(Xt, wtilde, y, n, jc, ir, d, lambda);
        
        // Print iterates and function values
        printf("  %ld       %+.15e       %ld       %.5e\n", t, fun_val,Kt,n_evals*nInverse);
        
        // Evaluate full gradient of the function
        compute_fun_fullgrad_sparse<R>(Xt, wtilde, y, n, d, jc, ir, vtilde);
        
        // Set wk(w0) <- wtilde
        wk = wtilde;
        
        // Get non-uniform sampling set
        Kt = discrete_dist(generator);
        
        /* Inner Iteration Starts Here */
        for(k=0;k<Kt;k++){
            
            // Get uniform sampling set
            ik = rand()%n;
   
            // nnz: number of "non-zero" coordinates
            nnz = jc[ik+1]-jc[ik];
            
            
            
            // Lazy update;
            lazy_update_S2GD<R>(wk, vtilde, last_seen, eta, lambdapar, lambdapar2, k,
                                ir, jc[ik], jc[ik+1]);
            
            
            
            // Compute first two parts of the stochastic gradient: fi_wk-fi_wtilde
            fi_wk = compute_fun_grad_sparse<R>(Xt+jc[ik], wk, y[ik], nnz, ir+jc[ik]);
            fi_wtilde = compute_fun_grad_sparse<R>(Xt+jc[ik], wtilde, y[ik], nnz,
                                                   ir+jc[ik]);
                
            for(j=jc[ik];j<jc[ik+1];j++){
                wk[ir[j]] -= eta*(fi_wk-fi_wtilde)*Xt[j];
            }
            
        }
        // Finish lazy update
        finish_lazy_update_S2GD(wk, vtilde, last_seen, eta, lambdapar, lambdapar2, Kt, d);
        
        /* Inner Iteration Ends Here */
        n_evals += Kt+n;
        
        w = wk;
        
    }
    
    
    /* Outer Iteration Ends Here */
    
    
    /* Print the Last Iteration */
    // Evaluate current function value
    fun_val = compute_fun_val_sparse<R>(Xt, w, y, n, jc, ir, d, lambda);
    
    // Print the last iteration
    printf("  %ld       %+.15e       %ld       %.5e\n", t, fun_val,Kt,n_evals*nInverse);
    
    
    // Print footer
    if (printheader){
    cout<<"================================================================="<<endl;
    }
}


/* SGD_PVRG */
template<class R>
void SGD_PVRG(R *Xt, std::vector<R> &w, R *y, long *jc, long *ir, long n, long d,
              long T, long m, R eta, R lambda, int printheader, int seed)
{
    double sparsity = (double)jc[n]/(d*n);   // Compute sparsity of the data
    R nInverse = 1.0/n;
    R mInverse = 1.0/m;
    
    long n_evals = 0;
    // Calculate lambdapar for proximal operator
    R lambdapar = 1.0/(1.0+lambda*eta);
    
    srand(seed);   // Set randomseed
    
    // Print information
    // Print header
    if (printheader){
    cout<<"================================================================="<<endl;
    cout<<"    Method: "<<"SGD_PVRG"<<endl;
    cout<<"================================================================="<<endl;
    cout<<"    Parameters:"<<endl;
    cout<<"         n: "<<n<<endl;
    cout<<"         d: "<<d<<endl;
    cout<<"         sparsity: "<<sparsity<<endl;
    cout<<"         m: "<<m<<endl;
    cout<<"         maxiter: "<<T<<endl;
    cout<<"         eta: "<<eta<<endl;
    cout<<"         lambda: "<<lambda<<endl;
    cout<<"         randomseed: "<<seed<<endl;
    cout<<"================================================================="<<endl;
    cout<<"  Iter.   Fun.Val.                     Eff.Passes"<<endl;
    cout<<"================================================================="<<endl;
    }
    
    // Declare and allocate memory for useful parameters
    R fun_val;
    std::vector<R> wtilde(d);  // wtilde
    std::vector<R> vtilde(d);  // full gradient vtilde
    std::vector<R> wk(d);  // inner iterate
    std::vector<R> vk(d);  //  stochastic gradient vk
    long ik;  // uniform sampling
    R fi_wk, fi_wtilde;
    
    // Loop indexes
    unsigned long t, k, i, j;
    
    
    /* Outer Iteration Starts Here */
    for(t=0;t<T;t++){
        
        // Initialization for wtilde, vtilde and w
        wtilde = w;
        // Set w=0, vtilde = 0
        std::fill(w.begin(),w.end(),0);
        std::fill(vtilde.begin(),vtilde.end(),0);
        
        // Evaluate current function value
        fun_val = compute_fun_val_sparse<R>(Xt, wtilde, y, n, jc, ir, d, lambda);
        
        // Print iterates and function values
        printf("  %ld       %+.15e       %.5e\n", t, fun_val,n_evals*nInverse);
        
        // Evaluate full gradient of the function
        compute_fun_fullgrad_sparse<R>(Xt, wtilde, y, n, d, jc, ir, vtilde);
        
        // Set wk(w0) <- wtilde
        wk = wtilde;
        
        /* Inner Iteration Starts Here */
        for(k=0;k<m;k++){
            // Get uniform sampling set
            ik = rand()%n;
            // Set vk = 0
            std::fill(vk.begin(),vk.end(),0);
            /* Compute Stochastic Gradient (single SGD) */
            // Compute difference of single function gradient: vk <- grad_fi(wk) -
            // grad_fi(wtilde)
            fi_wk = compute_fun_grad_sparse<R>(Xt+jc[ik], wk, y[ik],jc[ik+1]-jc[ik], ir+jc[ik]);
            fi_wtilde = compute_fun_grad_sparse<R>(Xt+jc[ik], wtilde, y[ik], jc[ik+1]-jc[ik],
                                         ir+jc[ik]);
            
            // Update stochastic gradient by vk <- vk + vtilde
            for(j=jc[ik];j<jc[ik+1];j++){
                vk[ir[j]] += (fi_wk-fi_wtilde)*Xt[j];
            }
            
            for(i=0;i<d;i++){
                vk[i] = vk[i]+vtilde[i];
            }
            
            
            // Update inner iterate and outer iterate
            for(i=0;i<d;i++){
                wk[i] -= eta*vk[i];  // Inner iterate update
                wk[i] = lambdapar*wk[i];   // Proximal operator
                w[i] += wk[i]*mInverse;  // Outer iterate update
            }
            
        }
        /* Inner Iteration Ends Here */
        n_evals += m+n;
        
    }
    /* Outer Iteration Ends Here */
    
    
    /* Print the Last Iteration */
    // Evaluate current function value
    fun_val = compute_fun_val_sparse<R>(Xt, w, y, n, jc, ir, d, lambda);
    
    // Print the last iteration
    printf("  %ld       %+.15e       %.5e\n", t, fun_val,n_evals*nInverse);
    
    
    // Print footer
    if (printheader){
    cout<<"================================================================="<<endl;
    }
    
}

/* SAG */
template<class R>
void SAG(R *Xt, std::vector<R> &w, R *y, long *jc, long *ir, long n, long d,
         long T, R eta, R lambda, int printheader, int seed)
{
    double sparsity = (double)jc[n]/(d*n);   // Compute sparsity of the data
    R nInverse = 1.0/n;
    
    // Calculate lambdapar for proximal operator
    R lambdapar = 1.0/(1.0+lambda*eta);
    R lambdapar2 = 0;
    if (lambdapar<1){
        lambdapar2 = 1.0/(1.0-lambdapar);
    }
    
    srand(seed);   // Set randomseed
    
    // Print information
    // Print header
    if (printheader){
    cout<<"================================================================="<<endl;
    cout<<"    Method: "<<"SAG"<<endl;
    cout<<"================================================================="<<endl;
    cout<<"    Parameters:"<<endl;
    cout<<"         n: "<<n<<endl;
    cout<<"         d: "<<d<<endl;
    cout<<"         sparsity: "<<sparsity<<endl;
    cout<<"         maxiter: "<<T<<endl;
    cout<<"         eta: "<<eta<<endl;
    cout<<"         lambda: "<<lambda<<endl;
    cout<<"         randomseed: "<<seed<<endl;
    cout<<"================================================================="<<endl;
    cout<<"  Iter.   Fun.Val.                     Eff.Passes"<<endl;
    cout<<"================================================================="<<endl;
    }
    
    // Declare and allocate memory for useful parameters
    R fun_val;
    long ik;  // uniform sampling
    R fgrad_i;
    std::vector<R> fgrad_i_old(n);  // inner iterate
    std::vector<R> fgrad(d);  // inner iterate
    std::vector<long> last_seen(d);  // record iterates of last seen in lazy update
    
    // Loop indexes
    unsigned long t;
    
    // Initialization
    std::fill(fgrad_i_old.begin(),fgrad_i_old.end(),0);
    std::fill(fgrad.begin(),fgrad.end(),0);
    std::fill(last_seen.begin(),last_seen.end(),0);
    
    /* Iteration Starts Here */
    for(t=0;t<T;t++){
        
        // Initialization for wtilde, vtilde and w
        // Set w=0
        
        // Print iterates and function values
        if(t%n==0){
            // Finish lazy update for proximal operator
            finish_lazy_update_S2GD(w, fgrad, last_seen, eta, lambdapar, lambdapar2, t, d);
            
            // Evaluate current function value
            fun_val = compute_fun_val_sparse<R>(Xt, w, y, n, jc, ir, d, lambda);
            printf("  %ld       %+.15e       %.5e\n", t, fun_val,t*nInverse);
        }
        
        // Get uniform sampling
        ik = rand()%n;
        
        // Lazy update for proximal operator;
        lazy_update_S2GD<R>(w, fgrad, last_seen, eta, lambdapar, lambdapar2, t,
                            ir, jc[ik], jc[ik+1]);
        

        /* Compute Stochastic Gradient (single SGD) */
        fgrad_i = compute_fun_grad_sparse<R>(Xt+jc[ik], w, y[ik],jc[ik+1]-jc[ik],
                                             ir+jc[ik]);
        
        // Update the aggregate gradient
        for(unsigned long j=jc[ik];j<jc[ik+1];j++){
            fgrad[ir[j]] += nInverse*(fgrad_i-fgrad_i_old[ik])*Xt[j];   // Update iterate
        }
        
        // Record old fgrad_i
        fgrad_i_old[ik] = fgrad_i;
        
    }
    // Finish lazy update for proximal operator
    finish_lazy_update_S2GD(w, fgrad, last_seen, eta, lambdapar, lambdapar2, T, d);
    
    /* Iteration Ends Here */
    
    
    /* Print the Last Iteration */
    // Evaluate current function value
    fun_val = compute_fun_val_sparse<R>(Xt, w, y, n, jc, ir, d, lambda);
    
    // Print the last iteration
    printf("  %ld       %+.15e       %.5e\n", t, fun_val,t*nInverse);
    
    
    // Print footer
    if (printheader){
    cout<<"================================================================="<<endl;
    }
}

/* SGD */
template<class R>
void SGD(R *Xt, std::vector<R> &w, R *y, long *jc, long *ir, long n, long d,
         long T, R eta, R lambda, int printheader, int seed)
{
    double sparsity = (double)jc[n]/(d*n);   // Compute sparsity of the data
    R nInverse = 1.0/n;
    
    srand(seed);   // Set randomseed
    
    // Calculate lambdapar for proximal operator
    R lambdapar = 1.0/(1.0+lambda*eta);
    // Print information
    // Print header
    if (printheader){
        cout<<"================================================================="<<endl;
        cout<<"    Method: "<<"SGD"<<endl;
        cout<<"================================================================="<<endl;
        cout<<"    Parameters:"<<endl;
        cout<<"         n: "<<n<<endl;
        cout<<"         d: "<<d<<endl;
        cout<<"         sparsity: "<<sparsity<<endl;
        cout<<"         maxiter: "<<T<<endl;
        cout<<"         eta: "<<eta<<endl;
        cout<<"         lambda: "<<lambda<<endl;
        cout<<"         randomseed: "<<seed<<endl;
        cout<<"================================================================="<<endl;
        cout<<"  Iter.   Fun.Val.                     Eff.Passes"<<endl;
        cout<<"================================================================="<<endl;
    }
    
    // Declare and allocate memory for useful parameters
    R fun_val;
    long ik;  // uniform sampling
    R fgrad_i;
    std::vector<long> last_seen(d);  // record iterates of last seen in lazy update
    
    // Loop indexes
    unsigned long t, j;
    
    // Initialization
    std::fill(last_seen.begin(),last_seen.end(),0);
    
    /* Iteration Starts Here */
    for(t=0;t<T;t++){
        
        // Initialization for wtilde, vtilde and w
        // Set w=0
        
        
        
        // Print iterates and function values
        if(t%n==0){
            // Finish lazy update for proximal operator
            finish_lazy_update_SGD(w, last_seen, lambdapar, t, d);
            
            // Evaluate current function value
            fun_val = compute_fun_val_sparse<R>(Xt, w, y, n, jc, ir, d, lambda);
            
            printf("  %ld       %+.15e       %.5e\n", t, fun_val,t*nInverse);
        }
        
        // Get uniform sampling
        ik = rand()%n;
        
        // Lazy update for proximal operator;
        lazy_update_SGD<R>(w,last_seen, lambdapar, t, ir, jc[ik], jc[ik+1]);
        
        /* Compute Stochastic Gradient (single SGD) */
        fgrad_i = compute_fun_grad_sparse<R>(Xt+jc[ik], w, y[ik],jc[ik+1]-jc[ik],
                                             ir+jc[ik]);
        
        // Update stochastic gradient by wk <- wk - eta*fgrad_i
        for(j=jc[ik];j<jc[ik+1];j++){
            w[ir[j]] -= eta*fgrad_i*Xt[j];   // Update iterate
        }
    }
    // Finish lazy update for proximal operator
    finish_lazy_update_SGD(w, last_seen, lambdapar, T, d);
    
    /* Iteration Ends Here */
    
    
    /* Print the Last Iteration */
    // Evaluate current function value
    fun_val = compute_fun_val_sparse<R>(Xt, w, y, n, jc, ir, d, lambda);
    
    // Print the last iteration
    printf("  %ld       %+.15e       %.5e\n", t, fun_val,t*nInverse);
    
    
    // Print footer
    if (printheader){
        cout<<"================================================================="<<endl;
    }
}


/* SGD+ */
template<class R>
void SGD_plus(R *Xt, std::vector<R> &w, R *y, long *jc, long *ir, long n, long d,
         long T, R eta, R lambda, int printheader, int seed)
{
    double sparsity = (double)jc[n]/(d*n);   // Compute sparsity of the data
    R nInverse = 1.0/n;
    
    srand(seed);   // Set randomseed
    
    // Calculate lambdapar for proximal operator
    R lambdapar = 1.0/(1.0+lambda*eta);
    // Print information
    // Print header
    if (printheader){
    cout<<"================================================================="<<endl;
    cout<<"    Method: "<<"SGD"<<endl;
    cout<<"================================================================="<<endl;
    cout<<"    Parameters:"<<endl;
    cout<<"         n: "<<n<<endl;
    cout<<"         d: "<<d<<endl;
    cout<<"         sparsity: "<<sparsity<<endl;
    cout<<"         maxiter: "<<T<<endl;
    cout<<"         eta: "<<eta<<endl;
    cout<<"         lambda: "<<lambda<<endl;
    cout<<"         randomseed: "<<seed<<endl;
    cout<<"================================================================="<<endl;
    cout<<"  Iter.   Fun.Val.                     Eff.Passes"<<endl;
    cout<<"================================================================="<<endl;
    }
    
    // Declare and allocate memory for useful parameters
    R fun_val;
    long ik;  // uniform sampling
    R fgrad_i;
    std::vector<long> last_seen(d);  // record iterates of last seen in lazy update
    
    // Loop indexes
    unsigned long t, j;
    
    R eta0=eta;
    
    // Initialization
    std::fill(last_seen.begin(),last_seen.end(),0);
    
    /* Iteration Starts Here */
    for(t=0;t<T;t++){
        
        // Initialization for wtilde, vtilde and w
        // Set w=0
        
        
        // Print iterates and function values
        if(t%n==0){
            
            eta = eta0/(1.0+t/n);
            lambdapar = 1.0/(1.0+lambda*eta);
            
            // Finish lazy update for proximal operator
            finish_lazy_update_SGD(w, last_seen, lambdapar, t, d);
            
            // Evaluate current function value
            fun_val = compute_fun_val_sparse<R>(Xt, w, y, n, jc, ir, d, lambda);
            
            printf("  %ld       %+.15e       %.5e\n", t, fun_val,t*nInverse);
        }
        
        // Get uniform sampling
        ik = rand()%n;
        
        // Lazy update for proximal operator;
        lazy_update_SGD<R>(w,last_seen, lambdapar, t, ir, jc[ik], jc[ik+1]);
        
        /* Compute Stochastic Gradient (single SGD) */
        fgrad_i = compute_fun_grad_sparse<R>(Xt+jc[ik], w, y[ik],jc[ik+1]-jc[ik],
                                             ir+jc[ik]);
        
        // Update stochastic gradient by wk <- wk - eta*fgrad_i
        for(j=jc[ik];j<jc[ik+1];j++){
            w[ir[j]] -= eta*fgrad_i*Xt[j];   // Update iterate
        }
    }
    // Finish lazy update for proximal operator
    finish_lazy_update_SGD(w, last_seen, lambdapar, T, d);
    
    /* Iteration Ends Here */
    
    
    /* Print the Last Iteration */
    // Evaluate current function value
    fun_val = compute_fun_val_sparse<R>(Xt, w, y, n, jc, ir, d, lambda);
    
    // Print the last iteration
    printf("  %ld       %+.15e       %.5e\n", t, fun_val,t*nInverse);
    
    
    // Print footer
    if (printheader){
    cout<<"================================================================="<<endl;
    }
}




/* FISTA */
template<class R>
void FGD(R *Xt, std::vector<R> &w, R *y, long *jc, long *ir, long n, long d,
         long T, R eta, R lambda, int printheader)
{
    double sparsity = (double)jc[n]/(d*n);   // Compute sparsity of the data
    // Calculate lambdapar for proximal operator
    R lambdapar = 1.0/(1.0+lambda*eta);
    // Print information
    // Print header
    if (printheader){
        cout<<"================================================================="<<endl;
        cout<<"    Method: "<<"FGD"<<endl;
        cout<<"================================================================="<<endl;
        cout<<"    Parameters:"<<endl;
        cout<<"         n: "<<n<<endl;
        cout<<"         d: "<<d<<endl;
        cout<<"         sparsity: "<<sparsity<<endl;
        cout<<"         maxiter: "<<T<<endl;
        cout<<"         eta: "<<eta<<endl;
        cout<<"         lambda: "<<lambda<<endl;
        cout<<"================================================================="<<endl;
        cout<<"  Iter.   Fun.Val.                     Eff.Passes"<<endl;
        cout<<"================================================================="<<endl;
    }
    
    // Declare and allocate memory for useful parameters
    R fun_val, fun_val_prev, Q, tk, t0, step, Ak,a, mu;
    std::vector<R> xk(d);
    std::vector<R> xk_prev(d);
    std::vector<R> vk(d);  //  stochastic gradient vk
    std::vector<R> dk(d);
    
    // Loop indexes
    unsigned long t, i;
    
    for (i=0;i<d;i++){
        xk[i]=w[i];
    }
    std::fill(dk.begin(),dk.end(),0);
    t0 = 1;
    Ak = 0;
    mu = 1/n;
    
    
    /* Iteration Starts Here */
    for(t=0;t<T;t++){
        
        // Set vk = 0
        std::fill(vk.begin(),vk.end(),0);
        
        // Evaluate current function value
        fun_val = compute_fun_val_sparse<R>(Xt, w, y, n, jc, ir, d, lambda);
        fun_val_prev = fun_val;
        
        //Print iterates and function values
        printf("  %ld       %+.15e       %ld\n", t, fun_val,t);
        
        // Evaluate full gradient of the function
        compute_fun_fullgrad_sparse<R>(Xt, w, y, n, d, jc, ir, vk);
        
        
        for(i=0;i<d;i++){
            xk_prev[i] = xk[i];  // Outer iterate update
        }
        while (1){
            a = 1+mu*Ak;
            a = eta*(a+sqrt((a+4*Ak)*a));
            // Update iterate
            for(i=0;i<d;i++){
                xk[i] = w[i]-eta*vk[i];  // Outer iterate update
            }
            // Proximal update
            compute_proximal_operator<R>(xk, lambdapar, d);
            
            fun_val = compute_fun_val_sparse<R>(Xt, xk, y, n, jc, ir, d, lambda);
            Q = fun_val_prev;
            
            for (i=0;i<d;i++){
                dk[i]=xk[i]-w[i];
            }
            
            for (i=0;i<d;i++){
                Q+=(xk[i]-w[i])*vk[i] + 0.5*(dk[i]*dk[i]/eta + lambda*xk[i]*xk[i]);
            }
            if (fun_val<=Q){break;}
            eta = eta*0.5;
            
        }
        tk = (1+sqrt(1+4*t0*t0))*0.5;
        step = (t0-1)/tk;
        for(i=0;i<d;i++)
        {
            w[i] = xk[i] + step*(xk[i]-xk_prev[i]);
        }
        t0 = tk;
        
    }
    /* Iteration Ends Here */
    
    
    /* Print the Last Iteration */
    // Evaluate current function value
    fun_val = compute_fun_val_sparse<R>(Xt, w, y, n, jc, ir, d, lambda);
    
    //Print the last iteration
    printf("  %ld       %+.15e       %ld\n", t, fun_val,t);
    
    
    // Print footer
    if (printheader){
        cout<<"================================================================="<<endl;
    }
    
}




