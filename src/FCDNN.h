class FCDNN
{
    //Atrib
    private:
	//Hidden layers
	//y 2D matrix, raws are the output value of each neuron and columns are the layers
	float **y;
	float **net;
	//W 3D matrix, raws are the neurons, columns are the values of each param and depth are the layers.
	float*** W;
	int nParams;
	int nLayers;
	int neuronsPerLayer;

	//Input layer
	//inputW 2D matrix, raws are the output value of each neuron and columns are the layers
	float **inputW;
	float *x;
	int inputXSize;
	int inputNParams;

	//Output layer
	float **outputW;
	float *output;
	float * outputNet;
	int nOutputNeurons;

    float learningRate;



   //methods
    public:
	FCDNN(float *x, int sizeOfx, int nLayers, int neuronsPerLayer, int nOutputNeurons, float learningRate);
	void Train(float **rawX, float **labels, int sampleSize);
    float ** Evaluate(float **rawX, int sampleSize);


    private:
	void InitializeX(float *x, int nParams);
	void InitializeW();
	void InitializeY();
	void InitializeNet();
	void InitializeInput(float *x, int sizeOfx);
	void InitializeInputW();
	void InitializeOutputW();
	void InitializeOutput();
	void FeedFordward(float *x);
    //t -> espected output
	void GradientBackPropagation(float *t, float **deltas, float *outputDeltas);
    void StochasticWeightsUpdate(float *x, float **deltas, float *outputDeltas);
	float (*ActivaitonFunciton)(float);
	float (*DActivaitonFunciton)(float);

	float * AddInputBias(float *x);

    float RandomNumber();

    void ShowWeights();


};
