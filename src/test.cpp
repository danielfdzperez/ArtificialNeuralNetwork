#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "FCDNN.h"


int main(int argc, char *argv[]){

    float **input = new float*[4];
    //for (int i = 0; i < 4; i++)
    //    input[i] = new int[2];
    input[0] = new float[2]{0,0};
    input[1] = new float[2]{1,0};
    input[2] = new float[2]{0,1};
    input[3] = new float[2]{1,1};
    //{{0,0},{1, 0},{0,1}, {1,1}};


    float **labels = new float*[4];
    labels[0] = new float[1]{0};
    labels[1] = new float[1]{0};
    labels[2] = new float[1]{0};
    labels[3] = new float[1]{1};
	
    FCDNN a = FCDNN(input[0], 2, 3, 2, 1, 0.5);
    a.Train(input, labels, 4);

    float **answer = a.Evaluate(input,4);
    for(int i = 0; i<4; i++)
        printf("%.2f/n", answer[i][0]);


    return EXIT_SUCCESS;


} 

