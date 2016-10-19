/*
====================================================================================
        Mini-Batch Proximal Semi-Stochastic Gradient Method
====================================================================================
  Author: Jie Liu
  Website: http://coral.ie.lehigh.edu/~jild13/
  Last Update: Oct 24, 2014
====================================================================================
  Description:
      Generate Program to Run mS2GD and Related Algorithms, including:
  S2GD (by Jakub Konecny and Peter Richtarik: http://arxiv.org/abs/1312.1666 ),
  SGD_PVRG (by Tong Zhang: http://arxiv.org/abs/1403.4699 ),
  SAG (by Mark Schmidt et al.:http://papers.nips.cc/paper/4633-a-stochastic-gradient-method-with-an-exponential-convergence-_rate-for-finite-training-sets.pdf ),
  SGD and FGD.
------------------------------------------------------------------------------------
  Input:
        argc    ~ int, number of arguments
       *argv    ~ const char*, options/arguments for program
------------------------------------------------------------------------------------
 Options for algorithms(-a):
         1   ~ mS2GD
         2   ~ S2GD
         3   ~ SGD_PVRG
         4   ~ SAG
         5   ~ SGD
         6   ~ SGD+
         7   ~ FGD
====================================================================================
 Copyright by Jie Liu, H.S. Mohler Laboratory, Lehigh University.
====================================================================================
 */

#include "read_binary.h"
#include "SGDs.h"
#include "command_line.h"
using namespace std;

#define fun_prec 31
#define grad_evals_prec 4

void SGDsExperiment(cl_param* param);

int main(int argc, const char *argv[]) {

    cl_param* param = new cl_param;
    command_line_argument(argc, argv, param);
    
    SGDsExperiment(param);
    
    delete param;
    return 0;
}

void SGDsExperiment(cl_param* param){

    /* Read data from libsvm data file */
    sparseStruct data;
    data = read_binary(param->filename);
	long *jc, *ir, d, n;
	double *Xdata, *label;
	label = data.labels;
	jc = data.jc;
	ir = data.ir;
	Xdata = data.Xt;
	d = data.nf;
    n = data.N;
    
    /* Set parameters for the algorithms */
    // Set iterations and size of mini-batches
	long m = param->m;   // number of inner iterations
	long b = param->b;   // size of mini-batches
    long maxiter = param->maxiter;   // number of outer iterations
    
	double eta = param->eta;   // constant stepsize
    double lambda = param->lambda;   // regularization parameter for L2 norm
    
    double muF = param->muF;   // strong convexity parameter for F
    double muR = 1.0/n;   // strong convexity parameter for R
    
    int printheader = param->printheader;   // print header
    
    // Set random seed
    int seed[2] = {param->randomgenerator, param->randomseed};
    int sseed = param->randomseed;
    
    // Initial point
    std::vector<double> w(d);
    std::fill(w.begin(),w.end(),0);
    
    // Switch for algorithms: mS2GD, S2GD, mSGD_PVRG, SGD_PVRG, SGD, FGD
    switch(param->method) {
            /* mS2GD */
        case 1:
            mS2GD_PVRG<double>(Xdata, w, label, jc, ir, n, d, maxiter, m, b, eta,
                               muF, muR, lambda, printheader, seed);
            break;
            
            /* S2GD */
        case 2:
            S2GD_PVRG<double>(Xdata, w, label, jc, ir, n, d, maxiter, m, eta, muF,
                              muR, lambda, printheader, seed);
            break;
        
            /* SGD_PVRG */
        case 3:
            SGD_PVRG<double>(Xdata, w, label, jc, ir, n, d, maxiter, m, eta, lambda,
                             printheader,sseed);
            break;
            
            /* SAG */
        case 4:
            SAG<double>(Xdata, w, label, jc, ir, n, d, maxiter, eta, lambda, printheader,
                        sseed);
            break;
            
            /* SGD */
        case 5:
            SGD<double>(Xdata, w, label, jc, ir, n, d, maxiter, eta, lambda, printheader,
                        sseed);
            break;
            
            /* SGD+ */
        case 6:
            SGD_plus<double>(Xdata, w, label, jc, ir, n, d, maxiter, eta, lambda, printheader,
                        sseed);
            break;
            
            /* FGD */
        case 7:
            FGD<double>(Xdata, w, label, jc, ir, n, d, maxiter, eta, lambda, printheader);
            break;
            
        default:
            printf("wrong input for option -a\n");
            exit(1);
    }
    
}





