/*
=====================================================================================
           Structure to Read libsvm Sparse Matrix
=====================================================================================
  Author: Jie Liu
  Website: http://coral.ie.lehigh.edu/~jild13/
  Last Update: Nov 18, 2014
=====================================================================================
  Usage:
        sparseStruct data;
        data = read_binary(const char *filename);
-------------------------------------------------------------------------------------
  Structure:
        *labels ~ double, labels from data, (d x 1)
        *jc    ~ long, indexes of first non-zero elements of
                columns of the data matrix, ((n+1) x 1)
        *ir    ~ long, row indexes of elements of the data matrix,0~nf-1
        *Xt    ~ double, sparse data matrix
         N     ~ long, number of data points
         nf    ~ long, number of features
-------------------------------------------------------------------------------------
  Input:
        *filename ~  filename of libsvm format data
=====================================================================================
 Copyright by Jie Liu, H.S. Mohler Laboratory, Lehigh University.
 ====================================================================================
 */
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <algorithm>
using namespace std;


// Define Sparse Matrix Structure
typedef struct{
	long *jc, *ir, nf, N;
	double *Xt, *labels;
} sparseStruct;


static char *line;
static long max_line_len;

// Function to read each line of libsvm format data
static char* readline(FILE *input)
{
	long len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len);
		len = (long) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}


// Read data into Sparse Data Structure
sparseStruct read_binary(const char *filename)
{
	char *endpt;
	FILE *prob = fopen(filename,"r");

	max_line_len = 1024;
	line = (char *) malloc(max_line_len*sizeof(char));

	long N=0,L=0;
	// Calculate the Dimension of the Data
	while(readline(prob) != NULL){
		char *label, *index, *value;
		N++;
		label = strtok(line," \t\n");
		while(1){
			index= strtok(NULL,":");
			value = strtok(NULL," \t");

			if(value == NULL)
				break;

			L++;
		}
	}
	rewind(prob);
    

	// Data Structure and Memory Allocation for the Structure
	sparseStruct data;
    data.jc = (long *) malloc((N+1)*sizeof(long));
	data.ir = (long *) malloc(L*sizeof(long));
	data.labels = (double *) malloc(N*sizeof(double));
	data.Xt = (double *) malloc(L*sizeof(double));
	data.jc[N] = L;
	data.nf = 0;

	// Loops to Assign Values to Sparse Structure
	long i=0;
	char *label, *index, *value;
	for(long n=0;n<N;n++){
		readline(prob);
		label = strtok(line," \t\n");  // Assign label
		data.labels[n] = strtod(label,&endpt);

		data.jc[n] = i;  // Assign jc
		while(1){
			index= strtok(NULL,":");
			value = strtok(NULL," \t");

			if(value == NULL)
				break;

			data.ir[i] = strtol(index,&endpt,10)-1; // Assign ir
			data.Xt[i] = strtod(value,&endpt);  // Assign Xt

			i++;
		}

        data.nf = std::max(data.nf,data.ir[i-1]);  // Number of Features
		data.N = N;  // Number of Data Points
	}
    data.nf++;

	fclose(prob);

	free(line);  // Free memory

	return data;
}


