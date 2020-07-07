#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define pi 3.14159265359


typedef struct{
    double **mu, **sigma;
    double *temps;
    double *acceptance, *swaps;
    int size, num_temps;
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
    
    //Faking an incorporation of a prior
    if(sigma<=0.0){
        return -INFINITY;
    }

    for(int i=0;i<data_size;i++){
        returnValue -= ((data[i]-mu)*(data[i]-mu))/(2*sigma*sigma);
    }
    return ((double) data_size) * -(1)*log(sqrt(2*sigma*sigma)) + returnValue;
}

returnParams allocateMemory(int N,int num_temps){
    returnParams output;

    //allocate pointer array (First dimension is Temp)
    output.mu=(double**)malloc(num_temps*sizeof(double*));
    output.sigma=(double**)malloc(num_temps*sizeof(double*));

    //allocate double arrays inside pointer array (Second dimension is iteration)
    for(int i=0; i<num_temps; i++){
        output.mu[i]=(double*)malloc(N*sizeof(double));
        output.sigma[i]=(double*)malloc(N*sizeof(double));
    }

    //allocating temps, setting the size (number of iterations), and number of temps
    output.temps = (double*)malloc(num_temps*sizeof(double));
    output.acceptance = (double*)malloc(num_temps*sizeof(double));
    output.swaps = (double*)malloc((num_temps-1)*sizeof(double));
    output.size = N;
    output.num_temps = num_temps;

    return output;
}

returnParams mcmc(double * data, int data_size, double * temps, int num_temps,
                  int N, double jumpsize, double swapChance){
    //allocating memory
    returnParams output = allocateMemory(N,num_temps);
    
    //To keep track of acceptance and swapping
    int accept[num_temps];
    int swaping[num_temps-1];

    //assign initial values
    for(int j = 0; j<num_temps; j++){
        //Assign temps
        output.temps[j]=temps[j];

        //Set initial values
        output.mu[j][0]=0;
        output.sigma[j][0]=0.1;

        //set acceptance to 1
        accept[j]=1;
        if(j<num_temps-1){
            swaping[j]=0;
        }
    }


    //Start MCMC----------------------------------
    //for each iteration...
    for(int i=1;i<N;i++){

        //for each chain...
        for(int j=0;j<num_temps;j++){
            
            //Probability of swaping
            if(i!=0 && (double) rand()/(RAND_MAX)<swapChance && j!=num_temps-1){
                
                //Chance for swapping chains
                //Getting variables needed (inefficient, but easy to read)
                double mu1 = output.mu[j][i-1];
                double mu2 = output.mu[j+1][i-1];
                double sigma1 = output.sigma[j][i-1];
                double sigma2 = output.sigma[j+1][i-1];
                double temp1 = output.temps[j];
                double temp2 = output.temps[j+1];

                //calculate the swap Hasting's in log
                double swapHTop = (1.0/temp1)*log_likelyhood(data,data_size,mu2,sigma2)+
                                  (1.0/temp2)*log_likelyhood(data,data_size,mu1,sigma1);
                double swapHBot = (1.0/temp2)*log_likelyhood(data,data_size,mu2,sigma2)+
                                  (1.0/temp1)*log_likelyhood(data,data_size,mu1,sigma1);
                double swapLogH = swapHTop-swapHBot;

                if(swapLogH>0 || swapLogH>= log((double)rand()/RAND_MAX)){

                    //accept the swap
                    //cold chain gets hot chain numbers
                    output.mu[j][i] = mu2;
                    output.sigma[j][i] = sigma2;

                    //hot chain gets cold chain numbers
                    output.mu[j+1][i] = mu1;
                    output.sigma[j+1][i]= sigma1;

                    swaping[j]++;

                }else{

                    //Not sure if I need to put anything here

                }
            }else{

                //Metropolis-Hastings
                //Get 2 random gaussian numbers for the parameters
                double * rando = randGaussian(0,jumpsize,2);
                double newMu = output.mu[j][i-1] + rando[0];
                double newSigma = output.sigma[j][i-1] + rando[1];
                
                //Free up memory allocated to rando
                free(rando);

                //calculate the hastings ratio with temperature scaling
                double oldLikely = (1.0/output.temps[j])*log_likelyhood(data, data_size, output.mu[j][i-1],output.sigma[j][i-1]);
                double newLikely = (1.0/output.temps[j])*log_likelyhood(data, data_size, newMu, newSigma);
                double logH = newLikely-oldLikely;
                
                if(logH>0 || logH>= log((double)rand()/RAND_MAX)){
                    
                    //accept the jump
                    output.mu[j][i] = newMu;
                    output.sigma[j][i] = newSigma;
                    accept[j]++;
                
                }else{
                    
                    //Not accepting the jump
                    output.mu[j][i]=output.mu[j][i-1];
                    output.sigma[j][i]=output.sigma[j][i-1];

                }

            }


        }
    }

    //Saving the swapping percentages and acceptance percentages
    for(int j=0; j<num_temps-1; j++){
        output.acceptance[j] = (double) accept[j]/N;
        output.swaps[j] = (double) swaping[j]/N;
    }

    //Since there is one less chain pairs than chains....
    //(i.e with 3 chains, there is two pairs, 1&2 and 2&3)
    output.acceptance[num_temps-1]=(double) accept[num_temps-1]/N;

    return output;
}


int main(){
    //random number generator
    srand(time(0));
    
    printf("Generating gaussian.\n");
    int data_size = 10000;
    double * data = randGaussian(1.0,0.6,data_size);
    /*
    double * data2 = randGaussian(6.0,0.1,data_size);

    for(int i=0; i<data_size; i+=2){
        data[i] = data2[i]; 
    }
    */

    printf("Finished generating gaussian.\n");

    double temps[] = {1.0,1.5,2.0,2.5,3.0,3.5};
    int num_temps = 6;

    printf("Starting MCMC.\n");
    returnParams out = mcmc(data,data_size,temps,num_temps,5000,.004,0.0);
    printf("Finished MCMC.\n");
    
    for(int i =0; i<num_temps;i++){
        printf("Acceptance of chain %i: %f\n",i,out.acceptance[i]);
    }
    

    FILE *f;
    f = fopen("gaussian.txt","w");
    for(int i=0;i<data_size;i++){
        fprintf(f,"%f\n",data[i]);
    }
    fclose(f);

    
    FILE *f1;
    f1 = fopen("mcmc_out.txt","w");
    for(int i=0;i<out.size;i++){
        fprintf(f1,"%f\t%f\n",out.mu[0][i],out.sigma[0][i]);
    }
    fclose(f1);
    
    return 0;
    
}