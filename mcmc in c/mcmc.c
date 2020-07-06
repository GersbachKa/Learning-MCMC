#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define pi 3.14159265359


typedef struct{
    double *mu, *sigma;
    double acceptance;
    int size;
}returnParams;


double boxMuller(void){
    double r1 = (double) rand()/(RAND_MAX);
    double r2 = (double) rand()/(RAND_MAX);
    return sqrt(-2*log(r1))*cos(2.0*pi*r2);
}


double * randGaussian(double mu, double sigma, int arr_size){
    double* arr = (double*)malloc(arr_size*sizeof(double));
    for(int i=0;i<arr_size;i++){
        arr[i]=sigma*boxMuller()+mu;
    }
    return arr;
}


double log_likelyhood(double * data, int data_size, double mu, double sigma){
    double returnValue = 0;
    for(int i=0;i<data_size;i++){
        returnValue -= ((data[i]-mu)*(data[i]-mu))/(2*sigma*sigma);
    }
    return ((double) data_size) * -(1)*log(sqrt(2*sigma*sigma)) + returnValue;
}

returnParams allocateMemory(int N){
    returnParams output;
    output.mu=(double*)malloc(N*sizeof(double));
    output.sigma=(double*)malloc(N*sizeof(double));
    output.size = N;
    return output;
}

returnParams mcmc(double * data, int data_size, int N, double jumpsize){
    returnParams output = allocateMemory(N);
    output.mu[0]=0;
    output.sigma[0]=0.1;
    int accept = 1;
    for(int i=1;i<N;i++){
        double * rando = randGaussian(0,jumpsize,2);
        double newMu = output.mu[i-1] + rando[0];
        double newSigma = output.sigma[i-1] + rando[1];
        free(rando);        
        double oldLikely = log_likelyhood(data, data_size, output.mu[i-1],output.sigma[i-1]);
        double newLikely = log_likelyhood(data, data_size, newMu, newSigma);
        double logH = newLikely-oldLikely;
        if(logH>0 || logH>= log((double)rand()/RAND_MAX)){
            output.mu[i] = newMu;
            output.sigma[i] = newSigma;
            accept++;
        }else{
            output.mu[i]=output.mu[i-1];
            output.sigma[i]=output.sigma[i-1];
        }
    }
    output.acceptance = (double) accept/N;
    return output;
}


int main(){
    //random number generator
    srand(time(0));
    
    printf("Generating gaussian.\n");
    int data_size = 10000;
    double * data = randGaussian(1.0,1.0,data_size);
    printf("Finished generating gaussian.\n");

    
    printf("Starting MCMC.\n");
    returnParams out = mcmc(data, data_size, 10000, .005);
    printf("Finished MCMC.\n");
    printf("Acceptance of: %f\n",out.acceptance);
    

    FILE *f;
    f = fopen("gaussian.txt","w");
    for(int i=0;i<data_size;i++){
        fprintf(f,"%f\n",data[i]);
    }
    fclose(f);

    
    FILE *f1;
    f1 = fopen("mcmc_out.txt","w");
    for(int i=0;i<out.size;i++){
        fprintf(f1,"%f\t%f\n",out.mu[i],out.sigma[i]);
    }
    fclose(f1);
    
    return 0;
    
}