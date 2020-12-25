#include <algorithm>    // std::copy
#include <vector>       // std::vector, std::begin, std::end
#include <cstring>      //memset
#include <cmath>        //exp
#include <cstdlib>      //rand
#include "FCDNN.h"

#include <stdio.h>

float Sigmoid(float x)
{
    return  1/(1+std::exp(-x));
}

float DSigmoid(float x)
{
    float r = Sigmoid(x);
    return r  * (1-r);
}

float FCDNN::RandomNumber()
{
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 
}

FCDNN::FCDNN(float *x, int sizeOfx, int nLayers, int neuronsPerLayer, int nOutputNeurons, float learningRate)
{
    this->nLayers = nLayers;
    this->neuronsPerLayer = neuronsPerLayer;
    this->nParams = neuronsPerLayer + 1; 
    this->nOutputNeurons = nOutputNeurons;
    this->inputXSize = sizeOfx;
    this->learningRate = learningRate;
    srand (static_cast <unsigned> (0));
    InitializeInput(x, sizeOfx);
    InitializeW();
    InitializeY();
    InitializeNet();
    InitializeOutputW();
    InitializeOutput();

    ActivaitonFunciton = &Sigmoid;
    DActivaitonFunciton = &DSigmoid;

}

//Input initialization
void FCDNN::InitializeInput(float *x, int sizeOfx)
{
    inputNParams = sizeOfx+1;
    //InitializeX(x);
    InitializeInputW();
}

//void FCDNN::InitializeX(float *x, int sizeOfX)
//{
//    //int sizeOfX = (*(&x + 1) - x);
//
//    //Add 1 place to the bias
//    this->x = new float[sizeOfX];
//
//    //for(int i = 0; i< sizeOfX; i++)
//    //    this->x[i] = x[i];
//   
//   std::copy(x, x+sizeOfX-1, this->x);
//   //Set bias to 1
//   this->x[sizeOfX-1] = 1;
//   
//
//}
void FCDNN::InitializeInputW()
{
    inputW = new float*[neuronsPerLayer];
    for(int i =0; i<neuronsPerLayer; i++)
        inputW[i] = new float [inputNParams];
    for(int i =0; i<neuronsPerLayer; i++)
        for(int j =0; j<inputNParams; j++)
            inputW[i][j] = RandomNumber();

}

////Hiden initialization
void FCDNN::InitializeW()
{
    W = new float**[nLayers];
    for(int i =0; i<nLayers; i++)
        W[i] = new float *[neuronsPerLayer];
    for(int i =0; i<nLayers; i++)
        for(int j =0; j<neuronsPerLayer; j++)
            W[i][j] = new float [nParams];

    for(int l=0; l<nLayers; l++)
        for(int i =0; i<neuronsPerLayer; i++)
            for(int j = 0; j < nParams; j++)
                 W[l][i][j] = RandomNumber();

}
void FCDNN::InitializeY()
{
    y = new float*[nLayers];
    for(int i =0; i<nLayers; i++)
	y[i] = new float [nParams];

    //Set bias to 1
    for(int i =0; i<nLayers; i++)
	y[i][nParams-1] = 1;
}
void FCDNN::InitializeNet()
{
    net = new float*[nLayers];
    for(int i =0; i<nLayers; i++)
	net[i] = new float [neuronsPerLayer];

    outputNet = new float[nOutputNeurons];
}
//
////Output initialization
void FCDNN::InitializeOutputW()
{
    outputW = new float*[nOutputNeurons];
    for(int i =0; i<nOutputNeurons; i++)
        outputW[i] = new float [nParams];
    for(int i =0; i<nOutputNeurons; i++)
        for(int j =0; j<nParams; j++)
            outputW[i][j] = RandomNumber();

}
//
void FCDNN::InitializeOutput()
{
    output = new float[nOutputNeurons];
}

void FCDNN::Train(float **rawX, float **labels, int sampleSize)
{
    //Bucle
    float *outputDeltas = new float[nOutputNeurons];
    float **deltas = new float*[nLayers]; 
    for(int l = 0; l<nLayers; l++)
        deltas[l] = new float[neuronsPerLayer];

    for (int i = 0; i < 10000; i++)
    {
        //printf("\n\r Iteracion %i", i);
        for(int s = 0; s<sampleSize; s++)
        {
            float *x = AddInputBias(rawX[s]);
            FeedFordward(x);

            //Test
            //printf("\n===================================================");
            //printf("\n Input [");
            //for(int z = 0; z<inputXSize+1; z++)
            //    printf("%f ", x[z]);
            //printf("]\n");

            //ShowWeights();
            //char *a;
            //if(inputW[0][0] > 10 || inputW[0][0] < -10)
            //    scanf("%s",a);


            //printf("\n y and nets");
            //for(int n=0; n<nLayers; n++){
            //    printf("\n layer %i => ",n);
            //    for(int m =0; m<neuronsPerLayer; m++){
            //        printf("y = %f ",y[n][m]);
            //        printf("net = %f ",net[n][m]);
            //    }
            //}
            //printf("\n");

            //printf("\n Output");
            //for(int z =0; z<nOutputNeurons; z++)
            //    printf("%f ", output[z]);
            //printf("\n");

            GradientBackPropagation(labels[s], deltas, outputDeltas);
            StochasticWeightsUpdate(x, deltas, outputDeltas);


            //printf("\n Output deltas");
            //for(int z =0; z<nOutputNeurons; z++)
            //    printf("%f ", outputDeltas[z]);

            //printf("\n Hiden deltas");
            //for(int l = 0; l<nLayers; l++){
            //    printf("\n layer %i => ",l);
            //    for(int m =0; m<neuronsPerLayer; m++)
            //        printf("%f ",deltas[l][m]);
            //}
            //printf("\n New weights");
            //ShowWeights();
            //printf("\n===================================================");

        }
    }


    //Gradien back propagation
    //wight updates
}


void FCDNN::ShowWeights()
{

    printf("\n");
    printf("Input weights\n");
    for(int i =0; i<neuronsPerLayer; i++)
        for(int j = 0; j < inputNParams; j++)
          printf("%f ",inputW[i][j]);
    printf("\n");

    printf("Hide weights\n");
    for(int l=1; l<nLayers; l++)
        for(int i =0; i<neuronsPerLayer; i++)
            for(int j = 0; j < nParams; j++)
                 printf("%f ",W[l][i][j]);
    printf("\n");

    printf("Output weights\n");
    for(int i =0; i<nOutputNeurons; i++)
    {
        for(int j = 0; j < nParams; j++)
            printf("%f ",outputW[i][j]);
    }
    printf("\n");
}


float ** FCDNN::Evaluate(float **rawX, int sampleSize)
{

    //for(int i =0; i<neuronsPerLayer; i++)
    //    for(int j = 0; j < inputNParams; j++)
    //      printf("%f ",inputW[i][j]);
    //printf("\n");

    //for(int l=1; l<nLayers; l++)
    //    for(int i =0; i<neuronsPerLayer; i++)
    //        for(int j = 0; j < nParams; j++)
    //             printf("%f ",W[l][i][j]);
    //printf("\n");
    //for(int i =0; i<nOutputNeurons; i++)
    //{
    //    for(int j = 0; j < nParams; j++)
    //        printf("%f ",outputW[i][j]);
    //}
    //printf("\n");




    float ** answer = new float*[sampleSize];
    for(int i = 0; i<sampleSize; i++)
        answer[i] = new float[nOutputNeurons];

    for(int s = 0; s<sampleSize; s++)
    {
        float *x = AddInputBias(rawX[s]);
        FeedFordward(x);

        //printf("[%f %f]  %f \n",x[0], x[1], output[0]);
        //for(int l=0; l<nLayers; l++)
        //    for(int i =0; i<neuronsPerLayer; i++){
        //        printf("l-%i  y = %f ",l,y[l][i]);
        //        printf("l-%i  net = %f ",l,net[l][i]);
        //    }

        //printf("\n");
        for(int i =0; i<nOutputNeurons; i++)
            answer[s][i] = output[i];
    }

    return answer;
    
}


//
void FCDNN::FeedFordward(float *x)
{
    //For each input neuron
    for(int i =0; i<neuronsPerLayer; i++)
    {
        float net = 0;
        for(int j = 0; j < inputNParams; j++){
            net += inputW[i][j] * x[j];
        }
        this->net[0][i] = net;
        //falta realizar una funcion
        //Add value to the y of the first layer
        y[0][i] = ActivaitonFunciton(net);
    }

    //For each hidden layer
    for(int l=1; l<nLayers; l++)
        for(int i =0; i<neuronsPerLayer; i++)
        {
            float net = 0;
            for(int j = 0; j < nParams; j++)
                net += W[l][i][j] * y[l-1][j];

	    this->net[l][i] = net;
            //falta realizar una funcion
            //Add value to the y of the first layer
            y[l][i] = ActivaitonFunciton(net);
        }

    //Output
    for(int i =0; i<nOutputNeurons; i++)
    {
        float net = 0;
        for(int j = 0; j < nParams; j++)
            net += outputW[i][j] * y[nLayers-1][j];
        outputNet[i] = net;
        //falta realizar una funcion en la salida
        //Add value to the y of the first layer
        output[i] = net;
    }
    
}

void FCDNN::GradientBackPropagation(float *t, float **deltas, float *outputDeltas)
{
    //printf("\n Back ");
    for(int i = 0; i < nOutputNeurons; i++)
    {
        outputDeltas[i] = (t[i] - output[i]);
        //printf("(%f - %f)  = %f\n",t[i], output[i],(t[i] - output[i]));
        //printf("(%f - %f) * %f = %f\n",t[i], output[i], DActivaitonFunciton(outputNet[i]),(t[i] - output[i]) * DActivaitonFunciton(outputNet[i]));
    }
    //printf("\n END Back ");

    for(int i = 0; i < neuronsPerLayer; i++)
    {
        float sum = 0;
        //No se tiene en cuenta el bias
        for(int j =0; j < nOutputNeurons; j++)
            sum += outputW[j][i] * outputDeltas[j];
        deltas[nLayers-1][i] = sum * DActivaitonFunciton(net[nLayers-1][i]);
    }

    for(int l = nLayers-1; l > 0; l--)
    {
        for(int i =0; i<neuronsPerLayer; i++)
        {
            float sum = 0;
            for(int j = 0; j < nParams; j++)
            {
                sum += W[l][i][j] * deltas[l][i];
            }
            deltas[l-1][i] = DActivaitonFunciton(net[l][i]) * sum;
        }
    }	
}

void FCDNN::StochasticWeightsUpdate(float *x, float **deltas, float *outputDeltas)
{
    //For each input neuron
    for(int i =0; i<neuronsPerLayer; i++)
    {
        //printf("-----------------------------\n");
        for(int j = 0; j < inputNParams; j++){
            //printf("%f + %f * %f * %f = %f",inputW[i][j],this->learningRate, deltas[0][i], x[j],inputW[i][j] + this->learningRate * deltas[0][i] * x[j]);
            inputW[i][j] = inputW[i][j] + this->learningRate * deltas[0][i] * x[j];
            //printf(" ## %f",inputW[i][j]);
            //printf("\n");
        }
            
    }
    //printf("################################\n");

    //For each hidden layer
    for(int l=1; l<nLayers; l++)
        for(int i =0; i<neuronsPerLayer; i++)
        {
            for(int j = 0; j < nParams; j++){
                 W[l][i][j]= W[l][i][j] + learningRate * deltas[l][i] * y[l-1][j];
                 //printf("%f %f %f\n",W[l][i][j], deltas[l][i], y[l-1][j]);
            }

        }
    //Output
    for(int i =0; i<nOutputNeurons; i++)
    {
        for(int j = 0; j < nParams; j++)
            outputW[i][j]= outputW[i][j] + learningRate * outputDeltas[i] * y[nLayers-1][j];
    }

}

//
float * FCDNN::AddInputBias(float *x)
{
    float * xWithBias;
    xWithBias = new float[inputXSize+1];

    //for(int i = 0; i< sizeOfX; i++)
    //    this->x[i] = x[i];
   
   std::copy(x, x+inputXSize, xWithBias);
   //Set bias to 1
   xWithBias[inputXSize] = 1;
   return xWithBias;

}
