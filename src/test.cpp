#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "FCDNN.h"


int main(int argc, char *argv[]){

    float **input = new float*[6];
    //for (int i = 0; i < 4; i++)
    //    input[i] = new int[2];
    input[0] = new float[2]{0,0};
    input[1] = new float[2]{1,0};
    input[2] = new float[2]{0,1};
    input[3] = new float[2]{1,1};
    input[4] = new float[2]{1,1};
    input[5] = new float[2]{1,1};
    //{{0,0},{1, 0},{0,1}, {1,1}};


    float **labels = new float*[6];
    labels[0] = new float[1]{0};
    labels[1] = new float[1]{0};
    labels[2] = new float[1]{0};
    labels[3] = new float[1]{1};
    labels[4] = new float[1]{1};
    labels[5] = new float[1]{1};


    float **test = new float*[5];
    //for (int i = 0; i < 4; i++)
    //    input[i] = new int[2];
    test[0] = new float[2]{0,0};
    test[1] = new float[2]{1,0};
    test[2] = new float[2]{0,1};
    test[3] = new float[2]{1,1};
    test[4] = new float[2]{1,1};
    //input, size, layers, neurons, outputneuron, alpha	
    FCDNN a = FCDNN(input[0], 2, 3, 2, 1, 0.2);
    a.Train(input, labels, 6);

    float **answer = a.Evaluate(test,5);
    for(int i = 0; i<5; i++){
        printf("[");
        for(int j = 0; j<2; j++)
        printf("%f ", test[i][j]);
        printf("] ");
        printf("%.2f\n", answer[i][0]);
    }


    return EXIT_SUCCESS;


} 

