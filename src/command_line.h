/*
 =================================================================================
     Command Line Structure for mS2GD
 =================================================================================
 Author: Jie Liu
 Website: http://coral.ie.lehigh.edu/~jild13/
 Last Update: Nov 24, 2014
 =================================================================================
 Description:
     Command Line Tools for Constructing Options for mS2GD in C++
 ---------------------------------------------------------------------------------
 Input:
       argc    ~ int, number of arguments
      *argv    ~ const char*, options/arguments for program
 =================================================================================
 Copyright by Jie Liu, H.S. Mohler Laboratory, Lehigh University.
 ====================================================================================
 */


using namespace std;
#include <iostream>
#include <fstream>
#include <stdlib.h>

#define max_len 1024

typedef struct {
    char* filename;
    long m, b, maxiter;
    double eta, lambda, muF, muR;
    int randomseed, randomgenerator, method, printheader;
} cl_param;


void command_line_argument(int argc, const char * argv[],
                           cl_param* param)
{
    
    // default values for input options
    param->m = 10000;
    param->b = 1;
    param->maxiter = 20;
    param->eta = 0.01;
    param->lambda = 1e-4;
    param->muF = 0;
    param->muR = param->lambda;
    param->filename = new char[max_len];
    param->method = 1;
    param->randomseed = 0;
    param->randomgenerator = 1;
    param->printheader = 1;
    
    // command options
    int i;
    for(i=1;i<argc;i++)
    {
        if(argv[i][0] != '-') break;
        if(++i>=argc) {
            printf("wrong input format\n");
            exit(1);
        }
        
        switch(argv[i-1][1])
        {
                /* (max) number of inner loops */
            case 'm':
                param->m = atol(argv[i]);
                break;
                
                /* mini-batch size */
            case 'b':
                param->b = atol(argv[i]);
                break;
                
                /* number of outer loops */
            case 'T':
                param->maxiter = atol(argv[i]);
                break;
                
                /* step-size */
            case 'e':
                param->eta = atof(argv[i]);
                break;
                
                /* convexity parameter for fucntion R */
            case 'r':
                param->muR = atof(argv[i]);
                break;
                
                /* regularization parameter */
            case 'l':
                param->lambda = atof(argv[i]);
                param->muR = param->lambda;
                break;
                
                /* convexity parameter for function F */
            case 'f':
                param->muF = atof(argv[i]);
                break;
                
                /* randomseed for mini-batch */
            case 's':
                param->randomseed = atoi(argv[i]);
                break;
                
                /* random generator for number of inner loops */
            case 'g':
                param->randomgenerator = atoi(argv[i]);
                break;
                
                /* algorithm/method to use */
            case 'a':
                param->method = atoi(argv[i]);
                break;
                
                /* print header(1) or not(0) */
            case 'p':
                param->printheader = atoi(argv[i]);
                break;
                
            default:
                printf("unknown option -%c\n", argv[i-1][1]);
                break;
        }
    }
    
    // decide if input of method is valid
    if(param->method < 1 || param->method > 7) {
        printf("wrong input for option -a\n");
        exit(1);
    }
    
    // decide if filename exists
    if(i>=argc) {
        printf("wrong input format\n");
        exit(1);
    }
    strcpy(param->filename,argv[i]);
    
}

