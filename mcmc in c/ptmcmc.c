#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include <omp.h> 
#define pi 3.14159265359

//For returnParams datatype-----------------------------------------------------------
typedef struct{
    char *name;
    double **parameters;
    double *temperatures;
    int *acceptance, *swaps;
    double (*likelihoodFunc)(double *, double *, int);
    int loglikely;
    int num_parameters, num_temps;
    double jump_scale;
    int currentIteration;
}allParams;

allParams allocateMemory(char *name, int num_parameters, int num_temperatures){
    allParams output;

    output.name=(char*)malloc(30*sizeof(char));
    
    strcat(output.name,name);

    //allocate pointer array (First dimension is temperatures)
    output.parameters=(double**)malloc(num_temperatures*sizeof(double*));
    output.acceptance=(int*)malloc(num_temperatures*sizeof(int));
    output.swaps=(int*)malloc(num_temperatures*sizeof(int));

    //allocate double arrays inside pointer array (Second dimension is parameter num)
    for(int i=0; i<num_temperatures; i++){
        output.parameters[i]=(double*)malloc(num_parameters*sizeof(double));
    }

    //allocating temperatures
    output.temperatures = (double*)malloc(num_temperatures*sizeof(double));

    output.num_parameters=num_parameters;
    output.num_temps = num_temperatures;
    output.currentIteration=0;

    return output;
}

void freeAllParams(allParams * toFree){
    free(toFree->name);
    for(int i=0;i<toFree->num_temps;i++){
       free(toFree->parameters[i]); 
    }
    free(toFree->parameters);
    free(toFree->acceptance);
    free(toFree->swaps);
    free(toFree->temperatures);
}

//Helper functions for MCMC----------------------------------------------------------

double boxMuller(void){
    double r1 = (double) rand()/(RAND_MAX);
    double r2 = (double) rand()/(RAND_MAX);
    return sqrt(-2*log(r1))*cos(2.0*pi*r2);
}

double randGaussian(double mu, double sigma){
    return sigma*boxMuller()+mu;
}

void writeToFile(allParams * current){
    //Each chain gets its own file
    for(int temp=0;temp<current->num_temps;temp++){
        char fileName[100]={'\0'};
        strcat(fileName,current->name);
        char endfile[20];
        sprintf(endfile,"_%i.txt",temp);
        strcat(fileName,endfile);

        FILE *f;
        f = fopen(fileName,"a");
        if (f==NULL){
            printf("File unable to be opened: %s",fileName);
            exit(EXIT_FAILURE);
        }
        char toAppend[1000]={'\0'};
        for(int i=0;i<current->num_parameters;i++){
            char numArr[20] = {'\0'};
            sprintf(numArr,"%f,",current->parameters[temp][i]);
            strcat(toAppend,numArr);
        }
        strcat(toAppend,"\n");
        fputs(toAppend,f);
        fclose(f);
    }
}

void printCurrentCondition(allParams * current, int max_n){
    int iter = current->currentIteration;
    double progress = (double) iter/max_n;
    printf("--------------------------------");
    printf("Current progress: %i/%i - %i%%\n",iter,max_n,(int)(progress*100));

    for(int temp=0;temp<current->num_temps;temp++){
        double chainAcceptance = (double) current->acceptance[temp]/iter;
        int swap = current->swaps[temp];
        printf("For chain %i - acceptance=%f%%, swaps=%i\n",temp,chainAcceptance*100,swap);
    }

}

void metropolisJump(allParams * current, double * data, int datasize, int temperature_num){
    double * paramsToJump = current->parameters[temperature_num];
    double newParams[current->num_parameters]; 
    double temperature = current->temperatures[temperature_num];

    for(int i=0;i<current->num_parameters;i++){
        newParams[i] = paramsToJump[i];
        newParams[i] += randGaussian(0,current->jump_scale);
    }

    double oldLikely = current->likelihoodFunc(paramsToJump,data,datasize);
    double newLikely = current->likelihoodFunc(newParams,data,datasize);
    double hastings,randAlpha;

    if(current->loglikely==1){
        oldLikely*=1/(temperature);
        newLikely*=1/(temperature);

        hastings = newLikely-oldLikely;
        randAlpha = log((double) rand()/(RAND_MAX));
    }else{
        oldLikely = pow(oldLikely,(1/temperature));
        newLikely = pow(newLikely,(1/temperature));

        hastings = newLikely/oldLikely;
        randAlpha = (double) rand()/(RAND_MAX);
    }

    if(hastings>randAlpha){
        current->acceptance[temperature_num]++;
        for(int i=0;i<current->num_parameters;i++){
            current->parameters[temperature_num][i]=newParams[i];
        }
    }
    writeToFile(current);
}

void proposeSwaps(allParams * current, double * data, int datasize){
    for(int temp=0;temp<current->num_temps-1;temp++){
        double like1= current->likelihoodFunc(current->parameters[temp],data,datasize);
        double temp1 = current->temperatures[temp];
        double like2= current->likelihoodFunc(current->parameters[temp+1],data,datasize);
        double temp2 = current->temperatures[temp+1];
        double hastings, randalpha;

        if(current->loglikely==1){
            hastings = temp1*(like2-like1)+temp2*(like1-like2);
            randalpha = log((double) rand()/(RAND_MAX));
        }else{
            hastings = (pow(like1,temp2)*pow(like2,temp1))/(pow(like1,temp1)*pow(like2,temp2));
            randalpha = (double) rand()/(RAND_MAX);
        }

        if(hastings>randalpha){
            current->swaps[temp]++;
            for(int i=0; i<current->num_parameters;i++){
                double toSwap = current->parameters[temp][i];
                current->parameters[temp][i] = current->parameters[temp+1][i];
                current->parameters[temp+1][i] = toSwap;
            }
        }
    }
    writeToFile(current);
}

allParams mcmc(char *name, int num_params, double *data, int datasize, int num_temps, double *tempArr, 
               double (*likelihoodFunc)(double*,double*,int), int loglikely, double jumpscale, int num_steps, int threads){
    
    //setting up the parameter structure
    allParams out = allocateMemory(name,num_params,num_temps);
    
    for(int i=0;i<num_temps;i++){
        out.temperatures[i]=tempArr[i];
    }

    out.likelihoodFunc = likelihoodFunc;
    out.loglikely = loglikely;
    out.jump_scale = jumpscale;

    for(int i=0;i<num_temps;i++){
        out.acceptance[i]=0;
        out.swaps[i]=0;
        for(int j=0;j<num_params;j++){
            out.parameters[i][j] = (double) rand()/(RAND_MAX);
        }
    }

    for(int outer=0;outer<num_steps;outer+=100){
        
        //loop over chains
        //#pragma omp parallel num_threads(threads);
        for(int temp=0;temp<num_temps;temp++){
            //99 iterations of metropolis
            for(int i=outer;(i%100)<99;i++){
                metropolisJump(&out,data,datasize,temp);
            }
        }

        proposeSwaps(&out,data,datasize);
        out.currentIteration+=100;
        printCurrentCondition(&out,num_steps);
    }

    return out;
}

double gaussianOneParam(double *params, double *data, int datasize){
    double x = params[0];
    double sigma = 1;
    double mu = 1;
    return (1.0/(sqrt(2.0*pi)*sigma))*exp(-pow(x-mu,2)/(2*pow(sigma,2)));
}

double gaussianTwoParam(double *params, double *data, int datasize){
    double mu = params[0];
    double sigma = params[1];

    double returnValue = 0;

    for(int i=0;i<datasize;i++){
        returnValue -= ((data[i]-mu)*(data[i]-mu))/(2*sigma*sigma);
    }
    return ((double) datasize) * -(1)*log(sqrt(2*sigma*sigma)) + returnValue;

}

int main(int argc, char* argv[]){

    /*
    int threads;
    printf("How many threads to run: ");
    scanf("%d",&threads);
    printf("Running with %i threads.\n",threads);
    */

    //random number generator
    srand(time(0));
    double temperature[] = {1.0,1.2,1.4,1.6,1.8};
    
    //One parameter
    allParams a = mcmc("gaussianOneParam",1,0,0,3,temperature,&gaussianOneParam,0,0.8,5000,1);
    freeAllParams(&a);

    //Two parameters
    double dataArr[1000];
    for(int i=0;i<1000;i++){ 
        dataArr[i] = randGaussian(1,1);
    }
    a = mcmc("gaussianTwoParam",2,dataArr,1000,3,temperature,&gaussianTwoParam,1,.002,1000,1);
    freeAllParams(&a);


    return 0;
    
}