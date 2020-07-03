#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define pi 3.14159265359
#define chainLength 50000


typedef struct{
    double mu[chainLength];
    double sigma[chainLength];
    double acceptance;
    int size;
}returnParams;


double boxMuller(void){
    double r1 = (double) rand()/(RAND_MAX);
    double r2 = (double) rand()/(RAND_MAX);
    return sqrt(-2*log(r1))*cos(2.0*pi*r2);
}


void randGaussian(double mu, double sigma, double * arr, int arr_size){
    if(arr[0]==0){
        for(int i=0;i<arr_size;i++){
            arr[i]+=sigma*boxMuller()+mu;
        }
    }else{
        for(int i=0;i<arr_size;i++){
            arr[i]=sigma*boxMuller()+mu;
        }
    }
}


double log_likelyhood(double * data, int data_size, double mu, double sigma){
    double returnValue = 0;
    for(int i=0;i<data_size;i++){
        returnValue -= ((data[i]-mu)*(data[i]-mu))/(2*sigma*sigma);
    }
    return ((double) data_size) * -(1)*log(sqrt(2*sigma*sigma)) + returnValue;
}


void mcmc(returnParams *output, double * data, int data_size, double jumpsize){
    output->mu[0]=0.0;
    output->sigma[0]=0.1;

    int accept = 1;
    for(int i=1;i<chainLength;i++){
        double rando[2]; 
        randGaussian(0,jumpsize,rando,2);
        double newMu = output->mu[i-1] + rando[0];
        double newSigma = output->sigma[i-1] + rando[1];
        double oldLikely = log_likelyhood(data, data_size, output->mu[i-1],output->sigma[i-1]);
        double newLikely = log_likelyhood(data, data_size, newMu, newSigma);
        double logH = newLikely-oldLikely;
        if(logH>0 || logH>= log((double)rand()/RAND_MAX)){
            output->mu[i] = newMu;
            output->sigma[i] = newSigma;
            accept++;
        }else{
            output->mu[i]=output->mu[i-1];
            output->sigma[i]=output->sigma[i-1];
        }
    }
    output->size = chainLength;
    output->acceptance = (double) accept/chainLength;
}


int main(){
    //random number generator
    srand(time(0));
    
    printf("Generating gaussian.\n");
    double data[10000] = {0};

    
    randGaussian(1.0,1.0,data,10000);
    printf("Finished generating gaussian.\n");

    
    printf("Starting MCMC.\n");
    returnParams out;
    mcmc(&out, data, 10000, .005);
    printf("Finished MCMC.\n");
    printf("Acceptance of: %f\n",out.acceptance);
    

    FILE *f;
    f = fopen("gaussian.txt","w");
    for(int i=0;i<10000;i++){
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